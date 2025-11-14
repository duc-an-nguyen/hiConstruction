import os
import json
import psycopg2
import psutil
import GPUtil
import time
import  paho.mqtt.client    as      mqtt
from    psycopg2.extras     import  Json
from    utility.logger      import  aiLogger

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
        PGDB   = os.getenv("PGDATABASE", "telemetry")
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
        
    # ===============================================
    # This function is used to connect
    # the MQTT
    # ------------------------------------------------
    def mqttConnect(self,client, userdata, flags, reason_code, properties):
        aiLogger.info(f"Connected rc: {reason_code}")
        if reason_code == 0:
            client.subscribe(self.mqttTopic, qos=0)
            aiLogger.info(f"Subscribed to: {self.mqttTopic}")
        else:
            aiLogger.warning("Not authorized or other connect error")
    
    # ===============================================
    # This function is used to handle
    # once we have a new message from edge
    # ------------------------------------------------
    def mqttMessage(self, client, userdata, msg):
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

        aiLogger.info(f"[CPU AND RAM] CPU={egdeCpu} RAM={edgeRam} of 7620MB")
        
        
        mqttLatencyMs           = None

        if timePub is not None:
            mqttLatencyMs           = (timeRecv - timePub) * 1000.0  # convert to ms
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
            self.insertRow(conn, msg.topic, data, self.sqlLatency)  # Save the previous latency time of SQL
            aiLogger.info("Inserted into PostgreSQL.")
            if self.sqlLatency is not None:
                aiLogger.info(f"SQL Latency stored: {self.sqlLatency:.2f} ms")
        except Exception as e:
            aiLogger.error("DB insert error:", e)


        if timePub is not None:
            self.sqlLatency = (time.time() - timePub)*1000.0
            self.sqlLatencyList.append(self.sqlLatency)
            aiLogger.info(f"[SQL LATENCY] messId={mesId} latency={self.sqlLatency:.2f} ms")

        # -------------------------------------------
        # For first message, set start time for 
        # throughput calc
        if self.firstRecvTime is None:
            self.firstRecvTime = timeRecv


        if mesId%500==0:
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
                CREATE TABLE IF NOT EXISTS telemetry (
                    id              BIGSERIAL PRIMARY KEY,
                    ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
                    topic           TEXT NOT NULL,
                    time            DOUBLE PRECISION,
                    rul             DOUBLE PRECISION,
                    reacHI          DOUBLE PRECISION,
                    sepHI           DOUBLE PRECISION,
                    stripHI         DOUBLE PRECISION,
                    sysHI           DOUBLE PRECISION,
                    sensors         JSONB,
                    cpu             DOUBLE PRECISION,
                    mem             DOUBLE PRECISION,
                    gpuLoad         DOUBLE PRECISION,
                    gpuTemp         DOUBLE PRECISION,
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
    def insertRow(self,pg_conn, topic: str, payload: dict, latency_ms):
        # known fields
        timestamp   = payload.get("time")
        rul         = payload.get("rul")
        reacHI      = payload.get("reacHI")
        sepHI       = payload.get("sepHI")
        stripHI     = payload.get("stripHI")
        sysHI       = payload.get("sysHI")

        cpu         = psutil.cpu_percent(interval=None)
        mem         = psutil.virtual_memory().percent
        gpuLoad, gpuTemp = self.getGpuStats()


        # keep everything else as sensors JSON (if not provided)
        sensors = payload.get("sensors")
        if sensors is None:
            core = {"Time", "rul","reacHI","sepHI","stripHI","sysHI","ts","schema","seq"}
            sensors = {k:v for k,v in payload.items() if k not in core}

        with pg_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO telemetry(latencyTime, time, topic, rul, reacHI, sepHI, stripHI, sysHI, cpu, mem, gpuLoad, gpuTemp, sensors)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (latency_ms, timestamp, topic, rul, reacHI, sepHI, stripHI, sysHI, cpu, mem, gpuLoad, gpuTemp, Json(sensors)))


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

        aiLogger.info("\n========== FINAL RESULTS ==========")
        aiLogger.info(f"Total messages received: {self.recvCount}")
        aiLogger.info(f"Total Retransmissions: {retraRatio}%")
        aiLogger.info(f"Delivery ratio: {deliveryRatio:.2f}%")
        aiLogger.info(f"Throughput: {throughput:.3f} msg/s")
        if self.sqlLatencyList:
            aiLogger.info(f"Average latency: {sum(self.sqlLatencyList)/len(self.sqlLatencyList):.2f} ms")
            aiLogger.info(f"p95 latency: {sorted(self.sqlLatencyList)[int(0.95*len(self.sqlLatencyList))-1]:.2f} ms")
        aiLogger.info(f"Goodput: {goodput:.4f} Mb/s")
        aiLogger.info(f"CPU Edge {sum(self.edgeCpu)/len(self.edgeCpu)}")
        aiLogger.info(f"RAM Edge {sum(self.edgeRam)/len(self.edgeRam)}")
        aiLogger.info("===================================")



# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# ------- MAIN FUNCTION -----------
# =================================
if __name__ == '__main__':
    iotSys = serverIoT()
    # -------------------------------
    # Create the client
    # -------------------------------
    pgConnect   = iotSys.postgreConnect()
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
    client.connect(iotSys.mqttHostBroker, iotSys.mqttPort, 60)
    client.loop_forever()
