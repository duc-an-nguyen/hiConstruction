# =============================================
# IMPORT
# =============================================
import  os
import  sys
import  threading
import  psutil
import  time
import  json
import  random
import  warnings

from    utility.config          import BufferConfig
from    utility.logger          import aiLogger
from    preProcessingDataTEP    import PreProcessData
from    model.tepSystemModel    import SystemModel

import  paho.mqtt.client        as mqtt
import  pandas                  as pd
import  numpy                   as np

# ===============================================
# This thread is used to communicate with the
# server and collect new weights, also the maintenace
# ===============================================
class EdgeCommunication():
    def __init__(self,config = None): 
        
        # ---------------------------
        # Loading config file
        # ---------------------------
        buffConfig                  = BufferConfig()
        self.preData                = PreProcessData(config=config)
        #----------------------------- 
        # Create thread and others
        #-----------------------------
        self.edge_communication     = threading.Thread(target=self.__edge_communication, daemon=False)
        
        self.__running              = False
        self.base_info              = {}
        self.utility_info           = {}

        #----------------------------
        # Condition object
        #----------------------------
        self.c_raw_edge_communication              = None
        self.c_processed_edge_communication        = threading.Condition()
        
        #----------------------------
        # Buffer
        #----------------------------
        self.raw_edge_communication_buffer         = None
        self.processed_edge_communication_buffer   = buffConfig.processedBuffer


        # ----------------------------
        # MQTT CONFIG
        # ----------------------------
        BROKER      = os.getenv("MQTT_HOST", "10.235.117.251")
        PORT        = int(os.getenv("MQTT_PORT", "1883"))
        TOPIC       = os.getenv("MQTT_TOPIC", "sensors/")
        SENSOR_ID   = os.getenv("SENSOR_ID", "edge1")
        USER        = os.getenv("MQTT_USER", "server")       
        PW          = os.getenv("MQTT_PW", "123456789")     

        self.mqttBroker = BROKER
        self.mqttPort   = PORT
        self.user       = USER
        self.passwd     = PW
        self.topic      = TOPIC

        self.client     = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="edge_sender_jetson1")
        # -------------------------------
        # Assign function of connection
        # -------------------------------
        if self.user and self.passwd:
            self.client.username_pw_set(self.user, self.passwd)


        self.client.on_disconnect = self.mqttDisconnect
        self.client.on_connect    = self.mqttConnect

        self.qos        = 1
        self.targetSize = 128
        

        self.connect()
        
    #==========================================
    # This function is used to start the thread
    #------------------------------------------         
    def start(self):
        self.__running = True
        self.edge_communication.start()
        
        
    #==========================================
    # This function is used to stop the thread
    #------------------------------------------   
    def stop(self):
        self.__running = False 
        
    #==========================================
    # This function expands layers buffer to 
    # multiple buffers connected to Ai Features
    #------------------------------------------     
    def __edge_communication(self):
        while True:
            try:
                # ----------------------------------------
                # wait till get new frame 
                # ----------------------------------------
                with self.c_raw_edge_communication:
                    if self.raw_edge_communication_buffer.empty():
                        self.c_raw_edge_communication.wait()
        
                        
                # ----------------------------------------
                # Get data from buffer
                # ----------------------------------------
                if not self.raw_edge_communication_buffer.empty():
                    # ------------------------------------
                    # Extract information from previous 
                    # thread
                    # ------------------------------------
                    base_info, utility_info = self.raw_edge_communication_buffer.get()

                    simTime                 = base_info["simTime"]
                    sensors                 = base_info["oriSignal"]
                    dataCollectionTime      = base_info["startFrameTime"]
                    idxFrame                = base_info["frameIdx"]
                    
                    flUpdate                = utility_info["flUpdate"]
                    anomaly                 = utility_info["anomaly"]
                    reacHI                  = utility_info["reacHI"]
                    sepHI                   = utility_info["sepHI"]
                    stripHI                 = utility_info["stripHI"]
                    sysHI                   = utility_info["sysHI"]
                    rul                     = utility_info["rul"]
                    
                    # ------------------------------------
                    # Pack the package to send
                    # ------------------------------------
                    cpu                     = psutil.cpu_percent(interval=None)
                    mem                     = psutil.virtual_memory().percent
                    # sensor_row              = sentData.iloc[idx].to_dict()
                    timePub                 = time.time()
                    
                    lastRowSensors          = sensors[0, -1, :]   # shape (45,)
                    # lastRowSensors          = pd.DataFrame([lastRowSensors], columns=self.preData.allWithoutLabelNoBias)
                    lastRowSensors          = dict(zip(self.preData.allWithoutLabelNoBias, lastRowSensors))

                    aiLogger.debug(f"The values of sensors is {lastRowSensors}")
                    aiLogger.debug(f"The type of sensors is {type(lastRowSensors)}")

                    aiLogger.debug(f"The value of HI is {reacHI.item()} type {type(reacHI.item())}")
                    aiLogger.debug(f"The value of RUL is {rul.item()} type {type(rul.item())}")

                    # print(f"-----------------------------------------------{simTime.shape}")
                    payload = { "mesId"                 : idxFrame,                            # unique ID for message
                                "dataCollectionTime"    : dataCollectionTime,
                                "timePub"               : timePub,
                                "simTime"               : simTime.item(),
                                "flUpdate"              : flUpdate,
                                "anomaly"               : anomaly,
                                "reacHI"                : reacHI.item(),
                                "sepHI"                 : sepHI.item(),
                                "stripHI"               : stripHI.item(),
                                "sysHI"                 : sysHI.item(),
                                "rul"                   : rul.item(),
                                "edgeCpu"               : cpu,
                                "edgeMem"               :mem,
                                **lastRowSensors 
                            }

                    jsonPayload         = json.dumps(payload)
                    payloadSizeBytes    = len(jsonPayload.encode("utf-8"))              # actual MQTT payload size
                    payloadSizeHuman    = f"{payloadSizeBytes/1024:.2f} KB"
                    aiLogger.info(f"The size of the sent package is {payloadSizeHuman}")
                    
                    self.client.publish(self.topic, jsonPayload,qos = 1)                       # Send the message
                    aiLogger.info(f"📤 Sent: {payload}")
                    
                
                if not self.__running:
                    break  


            except Exception as e:
                aiLogger.exception("__edge_communication Exception" + str(e))
    
    

        
    # =====================================================
    # This function is used for MQTT connect
    # -----------------------------------------------------
    def mqttConnect(self, client, userdata, flags, reason_code, properties):
        aiLogger.info(f"Connected to MQTT broker with result: {reason_code}")
        print(f"Connected to MQTT broker with result: {reason_code}")
        

    # =====================================================
    # This function is used to make a connection
    # -----------------------------------------------------
    def connect(self,):
        aiLogger.info(f"Connecting to MQTT broker at {self.mqttBroker}:{self.mqttPort} ...")
        self.client.connect(self.mqttBroker, self.mqttPort, 60)
        self.client.loop_start()
        aiLogger.info("Connected!!!")


    # =====================================================
    # This function is used to reconnect
    # -----------------------------------------------------
    def mqttDisconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        aiLogger.info(f"Disconnected with result code {reason_code}")
        # reason_code == 0 means clean disconnect
        if reason_code is not None and reason_code != 0:
            print("Unexpected disconnect. Trying to reconnect...")
            while True:
                try:
                    client.reconnect()   # no args
                    print("Reconnected!")
                    break
                except Exception as e:
                    print("Reconnect failed:", e)
                    time.sleep(5)  # wait then retry

    # =====================================================
    # This function is used to make payload to test
    # -----------------------------------------------------
    def make_payload(self,mesId, timePub, edgeCpu, edgeMem, targetSizeBytes):
        base = {
            "mesId": mesId,
            "timePub": timePub,
            "edgeCpu": edgeCpu,
            "edgeMem": edgeMem,
            "sensors": ""   # will fill later
        }

        # Step 1: encode base structure to get current size
        encoded         = json.dumps(base)
        currentSize     = len(encoded.encode("utf-8"))
        currentSize1    = len(encoded)
        aiLogger.info(f"The current size is {currentSize, currentSize1}")


        # Step 2: calculate how many bytes needed for padding
        padding_size = targetSizeBytes - currentSize

        if padding_size < 0:
            raise ValueError("Base payload already larger than target size.")

        # Step 3: fill padding
        base["sensors"] = "x" * padding_size

        return base

    # p64 = make_payload(mesId=1, edgeCpu=5.2, edgeMem=30.5, target_size_bytes=64)


 



                
                
            
        
        
        
        
        