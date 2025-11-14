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


# ===============================================
# LOAD THE DATA
# ===============================================

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

qos = 1


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
# ------------------------------------
# Environment variables
# ------------------------------------
BROKER      = os.getenv("MQTT_HOST", "10.219.155.251")
PORT        = int(os.getenv("MQTT_PORT", "1883"))
TOPIC       = os.getenv("MQTT_TOPIC", "sensors/")
SENSOR_ID   = os.getenv("SENSOR_ID", "edge1")
USER        = os.getenv("MQTT_USER", "server")       
PW          = os.getenv("MQTT_PW", "123456789")         

# ------------------------------------
# Create client (v2 API recommended)
# ------------------------------------
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)


# ------------------------------------
# Set username/password if provided
# ------------------------------------
if USER and PW:
    client.username_pw_set(USER, PW)

# ------------------------------------
# Connect to the broker
# ------------------------------------
print(f"Connecting to MQTT broker at {BROKER}:{PORT} ...")
client.connect(BROKER, PORT, 60)
print("Connected ✅")

idx = 0
idxCount = 0
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
    time.sleep(0.15)                                   #for AI task               
    payload = {
        "mesId"    : idxCount,                         # unique ID for message
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

    jsonPayload         = json.dumps(payload)
    payloadSizeBytes    = len(jsonPayload.encode("utf-8"))  # actual MQTT payload size
    payloadSizeHuman    = f"{payloadSizeBytes/1024:.2f} KB"
    print(f"The size of the sent package is {payloadSizeHuman}")

    # ----------------------------
    # Send the message
    client.publish(TOPIC, jsonPayload,qos=qos)
    print(f"📤 Sent: {payload}")
    
    time.sleep(1)                                    # send every 3 seconds
    idx+=1
    idxCount+=1
    if idx>= len(sentResults):
        idx = 0

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
