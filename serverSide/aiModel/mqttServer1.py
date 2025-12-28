import os
import json
import psycopg2
import psutil
import GPUtil
import time
import  paho.mqtt.client    as      mqtt
from    psycopg2.extras     import  Json
from    utility.logger      import  aiLogger
from    maintenance_activity import MaintenanceActivity

# =====================================
# This class is used to handle MQTT, SQL
# and display on Grafana
# =====================================
class serverIoT():
    def __init__(self): 
        # -------------------------------------------
        # Config environment variables for
        # MQTT
        # -------------------------------------------
        BROKER = os.getenv("MQTT_HOST", "mqtt-broker")
        PORT   = int(os.getenv("MQTT_PORT", "1883"))
        TOPIC  = os.getenv("MQTT_TOPIC", "sensors/#")
        USER   = os.getenv("MQTT_USER")
        PW     = os.getenv("MQTT_PW")

        self.mqttHostBroker = BROKER
        self.mqttPort       = PORT
        self.mqttTopic      = TOPIC
        self.mqttUser       = USER
        self.mqttPasswd     = PW
        
        # -------------------------------------------
        # Config environment variables for
        # PostgreSQL
        # -------------------------------------------
        PGHOST = os.getenv("PGHOST", "postgresql")
        PGPORT = int(os.getenv("PGPORT", "5432"))
        PGDB   = os.getenv("PGDATABASE", "healthIndicatorDatabase")
        PGUSER = os.getenv("PGUSER", "server")      
        PGPW   = os.getenv("PGPASSWORD", "123456789")  

        self.sqlHost    = PGHOST
        self.sqlPort    = PGPORT
        self.sqlDbName  = PGDB
        self.sqlUser    = PGUSER
        self.sqlPasswd  = PGPW

        # -------------------------------------------
        # The variables to measure the system
        # -------------------------------------------
        self.seenMesId          = set()
        self.duplicatedCount    = 0
        self.recvCount          = 0
        self.totalPayloadBytes  = 0
        self.sqlLatencyList     = []
        self.firstRecvTime      = None
        self.sqlLatency         = 0
        self.edgeCpu            = []
        self.edgeRam            = []   

        self.QoS                = 1


        self.testMQTTPerfomance = False
        self.maintenanceThr     = 0.999


        # ------------------------------------------
        # This value is used to inform to the maintenace
        # after each period. Starting value should be -1
        # because we need 0 at the begining
        # ------------------------------------------
        self.isMaintenance      = False
        self.infomMaintenance   = -1
        self.informPeriod       = 50


        # -------------------------------------------
        # AI for the server
        # -------------------------------------------
        if self.isMaintenance:
            self.maintenance = MaintenanceActivity()
            self.hi_repo,        \
            self.env_cfg,        \
            self.process_model,  \
            self.agent       = self.maintenance.preRunMaintenancePrediction()
        
    # ===============================================
    # This function is used to connect
    # the MQTT
    # ------------------------------------------------
    def mqttConnect(self,client, userdata, flags, reason_code, properties):
        aiLogger.info(f"Connected rc: {reason_code}")
        if reason_code == 0:
            client.subscribe(self.mqttTopic, qos=self.QoS)
            aiLogger.info(f"Subscribed to: {self.mqttTopic}")
        else:
            aiLogger.warning("Not authorized or other connect error")

    # ===============================================
    # This function is used to reconnect
    # the MQTT
    # ------------------------------------------------
    def mqttDisconnect(self,client, userdata, rc, *args, **kwargs):
        aiLogger.info(f"Disconnected with result code {rc}")
        # rc == 0 means clean disconnect
        if rc != 0:
            print("Unexpected disconnect. Trying to reconnect...")
            while True:
                try:
                    client.reconnect()
                    # client.reconnect(self.mqttHostBroker, self.mqttPort,60)
                    print("Reconnected!")
                    break
                except Exception as e:
                    print("Reconnect failed:", e)
                    time.sleep(5)  # wait then retry
    
    # ===============================================
    # This function is used to handle purely
    # once we have a new message from edge
    # ------------------------------------------------
    def mqttMessage(self, client, userdata, msg):

        aiLogger.info(f"The new message is: {msg.topic} {msg.payload.decode()}")

        # -------------------------------------------
        # Extract the data from message
        # -------------------------------------------
        try:
            self.receiData = json.loads(msg.payload.decode())
        except Exception as e:
            aiLogger.error("Bad JSON:", e)
            return

        # -------------------------------------------
        # Extract publish timestamp and 
        # message Id
        # -------------------------------------------
        mesId                   = self.receiData.get("mesId")
        timePub                 = self.receiData.get("timePub")

        egdeCpu                 = self.receiData.get("edgeCpu")
        edgeRam                 = self.receiData.get("edgeMem")
        self.edgeCpu.append(egdeCpu)
        self.edgeRam.append(edgeRam)

        # print(type(data), print(data.keys()))

        # -------------------------------------------
        # Check if we need maintenance or not, and 
        # insert into the data before writting into
        # SQL
        # -------------------------------------------
        if self.receiData.get("stripHI") < self.maintenanceThr:
            self.infomMaintenance                   += 1
            self.receiData["maintenanceComponent"]  = "Stripper"

        elif self.receiData.get("sepHI") < self.maintenanceThr:
            self.infomMaintenance                   += 1
            self.receiData["maintenanceComponent"]  = "Separator"

        elif self.receiData.get("reacHI") < self.maintenanceThr:
            self.infomMaintenance                   += 1
            self.receiData["maintenanceComponent"]  = "Reactor"
        else:
            self.receiData["maintenanceComponent"]  = None
            self.infomMaintenance                   = -1

            # If all component is ok, no need maintenance
            self.receiData["maintenance_action"]                = 0
            self.receiData["maintenance_level"]                 = 0
            self.receiData["maintenance_discounted_return"]     = 0
            self.receiData["maintenance_profit"]                = 0
            self.receiData["maintenance_pm_cost"]               = 0
            self.receiData["maintenance_fail_cost"]             = 0
            self.receiData["maintenance_stockout_cost"]         = 0
            self.receiData["maintenance_total_production"]      = 0
            self.receiData["maintenance_fail_count"]            = 0


        if self.isMaintenance and self.receiData["maintenanceComponent"] is not None and (self.infomMaintenance%self.informPeriod == 0):
            mainResults = self.maintenance.runMaintenancePrediction(    hi_repo            = self.hi_repo, 
                                                                        env_cfg             = self.env_cfg, 
                                                                        process_model       = self.process_model, 
                                                                        agent               = self.agent, 
                                                                        vaeSys_global_best  = self.maintenance.preBuiltAi.vaeSys_global_best, 
                                                                        env_id              = 4, 
                                                                        splitData           = "train",
                                                                        trajectIdx          = 1,
                                                                        atSimulationTime    = 200,
                                                                        businessProfile     = "balanced",
                                                                        predictionSteps     = 60) 

            bestPlanInfo  = mainResults['planner_best_info']

            aiLogger.error(f"key of the bestPlanInfo {bestPlanInfo.keys()}")

            self.receiData["maintenance_action"]               = bestPlanInfo['action']
            self.receiData["maintenance_level"]                = bestPlanInfo['level']
            self.receiData["maintenance_discounted_return"]     = bestPlanInfo['discounted_return']
            self.receiData["maintenance_profit"]               = bestPlanInfo['profit_sum']
            self.receiData["maintenance_pm_cost"]               = bestPlanInfo['pm_cost_sum']
            self.receiData["maintenance_fail_cost"]             = bestPlanInfo['pm_cost_sum']
            self.receiData["maintenance_stockout_cost"]         = bestPlanInfo['stockout_cost_sum']
            self.receiData["maintenance_total_production"]      = bestPlanInfo['production_sum']
            self.receiData["maintenance_fail_count"]            = bestPlanInfo['fail_count']

        aiLogger.info(f"[CPU AND RAM] CPU={egdeCpu} RAM={edgeRam} of 7620MB")
        
    
        # -------------------------------------------
        # Write into the database
        # -------------------------------------------
        conn = userdata["pg_conn"]
        try:
            self.insertRow(conn, msg.topic, self.receiData, self.sqlLatency)
            aiLogger.info("Inserted into PostgreSQL.")
        
        except Exception as e:
            aiLogger.error(f"DB insert error:{e}")


        if timePub is not None:
            self.sqlLatency = (time.time() - timePub)*1000.0
            self.sqlLatencyList.append(self.sqlLatency)
            aiLogger.info(f"[SQL LATENCY] messId={mesId} latency={self.sqlLatency:.2f} ms")

        


    # ===============================================
    # This function is used to handle
    # once we have a new message from edge and measure
    # MQTT perfomance
    # ------------------------------------------------
    def mqttMessageWithMQTTMeasurement(self, client, userdata, msg):
        timeRecv                = time.time()           # capture arrival ASAP
        aiLogger.info(f"The new message is: {msg.topic} {msg.payload.decode()}")

        # -------------------------------------------
        # Extract the data from message
        # -------------------------------------------
        try:
            data = json.loads(msg.payload.decode())
        except Exception as e:
            aiLogger.error("Bad JSON:", e)
            return

        # -------------------------------------------
        # Extract publish timestamp and 
        # message Id
        # -------------------------------------------
        mesId                   = data.get("mesId")
        timePub                 = data.get("timePub")

        egdeCpu                 = data.get("edgeCpu")
        edgeRam                 = data.get("edgeMem")
        self.edgeCpu.append(egdeCpu)
        self.edgeRam.append(edgeRam)

        # aiLogger.info(f"The ")

        aiLogger.info(f"[CPU AND RAM] CPU={egdeCpu} RAM={edgeRam} of 7620MB")
        
        # ---------------------------------------------
        # Count and check duplication if self.testMQTTPerfomance
        if self.testMQTTPerfomance:
            mqttLatencyMs           = None

            if timePub is not None:
                mqttLatencyMs       = (timeRecv - timePub) * 1000.0  # convert to ms
                aiLogger.info(f"[MQTT LATENCY] messId={mesId} latency={mqttLatencyMs:.2f} ms")
        
            self.recvCount          += 1
            payloadBytes            = len(msg.payload)
            self.totalPayloadBytes  += payloadBytes
            # -------------------------------
            # Check duplicated items
            if mesId in self.seenMesId:
                self.duplicatedCount += 1
                aiLogger.info(f"[DUPLICATE] seq={mesId}")
                return
            else:
                self.seenMesId.add(mesId)
        
        # -------------------------------------------
        # Write into the database
        # -------------------------------------------
        conn = userdata["pg_conn"]
        try:
            # self.insertRowDiffSize(conn, msg.topic, data, self.sqlLatency)  # Save the previous latency time of SQL
            self.insertRow(conn, msg.topic, data, self.sqlLatency)
            aiLogger.info("Inserted into PostgreSQL.")
            if self.sqlLatency is not None:
                aiLogger.info(f"SQL Latency stored: {self.sqlLatency:.2f} ms")
        except Exception as e:
            aiLogger.error(f"DB insert error:{e}")


        if timePub is not None:
            self.sqlLatency = (time.time() - timePub)*1000.0
            self.sqlLatencyList.append(self.sqlLatency)
            aiLogger.info(f"[SQL LATENCY] messId={mesId} latency={self.sqlLatency:.2f} ms")

        # -------------------------------------------
        # For first message, set start time for 
        # throughput calc
        if self.testMQTTPerfomance:
            if self.firstRecvTime is None:
                self.firstRecvTime = timeRecv
        
            if mesId%249==0:
                self.printFinalResults()
                self.firstRecvTime = timeRecv




    # ===============================================
    # This function is used to connect
    # with PostgreSQL
    # ------------------------------------------------
    def postgreConnect(self,):
        conn = psycopg2.connect(host    = self.sqlHost, 
                                port    = self.sqlPort,
                                dbname  = self.sqlDbName, 
                                user    = self.sqlUser, 
                                password= self.sqlPasswd)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS hi_database (
                    id                      BIGSERIAL PRIMARY KEY,
                    ts                      TIMESTAMPTZ NOT NULL DEFAULT now(),
                    topic                   TEXT NOT NULL,
                    mesId                   DOUBLE PRECISION,
                    simTime                 DOUBLE PRECISION,
                    reacHI                  DOUBLE PRECISION,
                    sepHI                   DOUBLE PRECISION,
                    stripHI                 DOUBLE PRECISION,
                    sysHI                   DOUBLE PRECISION,
                    rul                     DOUBLE PRECISION,
                    mainComp                TEXT,
                    sensors                 JSONB,
                    cpu                     DOUBLE PRECISION,
                    mem                     DOUBLE PRECISION,
                    gpuLoad                 DOUBLE PRECISION,
                    gpuTemp                 DOUBLE PRECISION,
                    latencyTime             DOUBLE PRECISION
                );
            """)
        
        return conn
    
    
    def postgreConnectDiffSize(self,):
        conn = psycopg2.connect(host    = self.sqlHost, 
                                port    = self.sqlPort,
                                dbname  = self.sqlDbName, 
                                user    = self.sqlUser, 
                                password= self.sqlPasswd)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS hi_database_simple (
                    id              BIGSERIAL PRIMARY KEY,
                    ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
                    topic           TEXT NOT NULL,
                    time            DOUBLE PRECISION,
                    sensors         JSONB,
                    edgeCpu         DOUBLE PRECISION,
                    edgeMem         DOUBLE PRECISION,
                    latencyTime     DOUBLE PRECISION
                );
            """)
        
        return conn
    
    # ===============================================
    # This functions is used to get GPU
    # state
    # ------------------------------------------------
    def getGpuStats(self,):
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return None, None   # no GPU present
            # choose first GPU (or aggregate if you prefer)
            g = gpus[0]
            load = g.load * 100.0 if g.load is not None else None
            temp = g.temperature if g.temperature is not None else None
            return load, temp
        except Exception:
            # GPUtil/NVML not working — make fields NULL
            return None, None

    # ===============================================
    # This functions is used to insert
    # rows into SQL
    # ------------------------------------------------
    def insertRow(self,pg_conn, topic: str, payload: dict, latencyTime):
        
        # known fields
        mesId       = payload.get("mesId")
        simTime     = payload.get("simTime")
        rul         = payload.get("rul")
        reacHI      = payload.get("reacHI")
        sepHI       = payload.get("sepHI")
        stripHI     = payload.get("stripHI")
        sysHI       = payload.get("sysHI")
        mainComp    = payload.get("maintenanceComponent")


        cpu         = psutil.cpu_percent(interval=None)
        mem         = psutil.virtual_memory().percent
        gpuLoad, gpuTemp = self.getGpuStats()


        # keep everything else as sensors JSON (if not provided)
        sensors = payload.get("sensors")
        if sensors is None:
            core = {"Time", "rul","reacHI","sepHI","stripHI","sysHI","ts","schema","seq", "maintenanceComponent"}
            sensors = {k:v for k,v in payload.items() if k not in core}

        try:
            with pg_conn.cursor() as cur:
                cur.execute(""" INSERT INTO hi_database( topic, 
                                                        mesId, 
                                                        simTime, 
                                                        reacHI, 
                                                        sepHI, 
                                                        stripHI, 
                                                        sysHI,
                                                        rul, 
                                                        mainComp, 
                                                        sensors,  
                                                        cpu, 
                                                        mem, 
                                                        gpuLoad, 
                                                        gpuTemp, 
                                                        latencyTime)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""", 
                                                        (topic, 
                                                        mesId, 
                                                        simTime, 
                                                        reacHI, 
                                                        sepHI, 
                                                        stripHI, 
                                                        sysHI, 
                                                        rul, 
                                                        mainComp,
                                                        Json(sensors), 
                                                        cpu, 
                                                        mem, 
                                                        gpuLoad, 
                                                        gpuTemp, 
                                                        latencyTime))
                aiLogger.warning("Wrote into the databased successfully!")

        except Exception as e:
            aiLogger.error("PostgreSQL insert failed")
            aiLogger.error(f"Error: {e}")
            aiLogger.error(f"mesId={mesId}, topic={topic}")

    # ===============================================
    # This functions is used to insert
    # rows into SQL
    # ------------------------------------------------
    def insertRowDiffSize(self, pg_conn, topic: str, payload: dict, latency_ms):
        edgeCpu  = payload.get("edgeCpu")
        edgeMem  = payload.get("edgeMem")
        sensors  = payload.get("sensors")   # this is your padding string

        # Store sensors as JSONB; wrap in dict for flexibility
        sensors_json = {"sensors": sensors} if sensors is not None else None

        with pg_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO healthIndicatorDatabase_simple(
                    latencyTime, topic, edgeCpu, edgeMem, sensors
                )
                VALUES (%s, %s, %s, %s, %s)
            """, (latency_ms, topic, edgeCpu, edgeMem, Json(sensors_json)))


    # ===============================================
    # This functions is used to print 
    # final results to measure
    # ------------------------------------------------
    def printFinalResults(self,):
        totalTime      = time.time() - self.firstRecvTime if self.firstRecvTime else 1
        deliveryRatio  = 100 * len(self.seenMesId) / (max(self.seenMesId)+1)            # approximate total sent
        throughput     = len(self.seenMesId) / totalTime
        goodput        = (8 * self.totalPayloadBytes) / (totalTime * 1e6)  # Mb/s
        retraRatio     = 100*self.duplicatedCount/self.recvCount

        aiLogger.warning("\n========== FINAL RESULTS ==========")
        aiLogger.warning(f"Total messages received: {self.recvCount}")
        aiLogger.warning(f"Total Retransmissions: {retraRatio}%")
        aiLogger.warning(f"Delivery ratio: {deliveryRatio:.2f}%")
        aiLogger.warning(f"Throughput: {throughput:.3f} msg/s")
        if self.sqlLatencyList:
            aiLogger.warning(f"Average latency: {sum(self.sqlLatencyList)/len(self.sqlLatencyList):.2f} ms")
            aiLogger.warning(f"p95 latency: {sorted(self.sqlLatencyList)[int(0.95*len(self.sqlLatencyList))-1]:.2f} ms")
        aiLogger.warning(f"Goodput: {goodput:.4f} Mb/s")
        aiLogger.warning(f"CPU Edge {sum(self.edgeCpu)/len(self.edgeCpu)}")
        aiLogger.warning(f"RAM Edge {sum(self.edgeRam)/len(self.edgeRam)}")
        aiLogger.warning("===================================")



# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# ------- MAIN FUNCTION -----------
# =================================
if __name__ == '__main__':
    iotSys = serverIoT()
    # -------------------------------
    # Create the client
    # -------------------------------
    pgConnect   = iotSys.postgreConnect()           # postgreSQL connection
    client      = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, userdata={"pg_conn": pgConnect}) 
    
    # -------------------------------
    # Create user and password
    # -------------------------------
    if iotSys.mqttUser and iotSys.mqttPasswd:
        client.username_pw_set(iotSys.mqttUser, iotSys.mqttPasswd)

    # -------------------------------
    # Connect and receive the messages
    # -------------------------------
    print(f"Connecting to {iotSys.mqttHostBroker}:{iotSys.mqttPort} ...")
    client.on_connect = iotSys.mqttConnect
    client.on_message = iotSys.mqttMessage
    client.on_disconnect = iotSys.mqttDisconnect
    client.connect(iotSys.mqttHostBroker, iotSys.mqttPort, 30)
    client.loop_forever()
