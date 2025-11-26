# ===============================================
#   IMPORT
# ===============================================
import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mainEdge')))
import psutil

import time
import json
import random
import warnings
import paho.mqtt.client     as mqtt
import pandas               as pd
import numpy                as np
from utility.logger         import aiLogger
from preProcessingDataTEP   import PreProcessData
# from prettytable            import PrettyTable
# from keras.optimizers       import Adam
# from utility.matpilotDraw   import MatPlot
# from sklearn.preprocessing  import MinMaxScaler
# import tensorflow           as tf
# import matplotlib.pyplot    as plt


class EdgeClient():
    def __init__(self):
        BROKER      = os.getenv("MQTT_HOST", "10.219.155.251")
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
    # This function is used to run the while loop to send data
    def run(self, sentData, sentResults, qos=1):
        
        idx         = 0
        idxCount    = 0
        # ------------------------------------
        # Send signal
        # ------------------------------------
        while True:
            # ----------------------
            # Convert sensor row to 
            # plain dictionary
            cpu         = psutil.cpu_percent(interval=None)
            mem         = psutil.virtual_memory().percent
            sensor_row = sentData.iloc[idx].to_dict()
            timePub    = time.time()
            time.sleep(0.15)                                            # for AI task       

            payload = { "mesId"  : idxCount,                            # unique ID for message
                        "timePub": timePub,
                        "reacHI" : sentResults["reacHI"][idx],
                        "sepHI"  : sentResults["sepHI"][idx],
                        "stripHI": sentResults["stripHI"][idx],
                        "sysHI"  : sentResults["sysHI"][idx],
                        "rul"    : sentData["demoRUL"][idx],
                        "edgeCpu": cpu,
                        "edgeMem" :mem,
                        **sensor_row 
                    }

            # payload =  self.make_payload(mesId           =   idxCount, 
            #                              timePub         =   timePub,
            #                              edgeCpu         =   cpu,
            #                              edgeMem         =   mem,
            #                              targetSizeBytes =   self.targetSize
            #                              )

            jsonPayload         = json.dumps(payload)
            payloadSizeBytes    = len(jsonPayload.encode("utf-8"))  # actual MQTT payload size
            payloadSizeHuman    = f"{payloadSizeBytes/1024:.2f} KB"
            aiLogger.info(f"The size of the sent package is {payloadSizeHuman}")

            # ----------------------------
            # Send the message
            self.client.publish(self.topic, jsonPayload,qos=qos)
            print(f"📤 Sent: {payload}")
            
            time.sleep(1)                                    # send every 3 seconds
            idx+=1
            idxCount+=1
            if idx>= len(sentResults):
                idx = 0
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

# ===============================================
# ------------ MAIN FUNCTION --------------------
# ===============================================
if __name__ == '__main__':

    # ===========================================
    # Data preprocessing
    # ===========================================
    # ---------------------------
    # Loading config file
    # ---------------------------
    with open('utility/modelConfig.json', 'r') as configFile:
        configData = json.load(configFile)
    config = configData["tep"]  

    # ---------------------------
    # Set debug level and 
    # ---------------------------
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
    warnings.filterwarnings('ignore') 
    np.random.seed(1)
    random.seed(1)

    preData             = PreProcessData(config=config)
    rawData             = preData.loadDataFromH5Format(
                                                    name             = "data/teps.h5",
                                                    case             = "all", 
                                                    reduceSampleRate = True, 
                                                    removeHigherThr  = False)

    withRULData         = preData.addRULAndEnvironment(rawData)
    caseData            = withRULData[4]
    caseData["demoRUL"] = np.where( caseData['rul'] > 3,
                                    caseData['rul'] + np.random.uniform(-3, 3, size=len(caseData)),
                                    caseData['rul']  # keep original value if <= 3
                                )

    hiResults           = df = pd.read_excel("data/allHIs.xlsx")

    sentData            = caseData[caseData["Id"] == 17].reset_index(drop = True)
    sentResults         = hiResults[hiResults["id"] == 17].reset_index(drop = True)


    sentData            = sentData[:len(sentResults)]

    aiLogger.info(f"The size of sentData is {sentData.shape}, Results is {sentResults.shape}")


    # ===============================================
    #                   MQTT
    # ===============================================
    edgeSevice = EdgeClient()
    edgeSevice.connect()
    edgeSevice.run(sentData, sentResults, qos = edgeSevice.qos)





# docker build -t edge-sender .

# docker run --rm \
#   -e MQTT_HOST=192.168.68.251 \
#   -e MQTT_PORT=1883 \
#   -e MQTT_USER=server \
#   -e MQTT_PW=123456789 \
#   -e SENSOR_ID=edge1 \
#   -e MQTT_TOPIC='sensors/temperature' \
#   edge-sender



# docker run -it -v $PWD/mainEdge/data/:/app/data/  -e MQTT_HOST=10.219.155.251   -e MQTT_PORT=1883   -e MQTT_USER=server   -e MQTT_PW=123456789   -e SENSOR_ID=edge1   -e MQTT_TOPIC='sensors'   edge-sender-with_data
