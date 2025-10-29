import os
import time
import json
import random
import paho.mqtt.client as mqtt

# Environment variables
BROKER      = os.getenv("MQTT_HOST", "192.168.68.251")
PORT        = int(os.getenv("MQTT_PORT", "1883"))
TOPIC       = os.getenv("MQTT_TOPIC", "sensors/temperature")
SENSOR_ID   = os.getenv("SENSOR_ID", "edge1")
USER        = os.getenv("MQTT_USER")        # <-- new
PW          = os.getenv("MQTT_PW")          # <-- new

# Create client (v2 API recommended)
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

# Set username/password if provided
if USER and PW:
    client.username_pw_set(USER, PW)

print(f"Connecting to MQTT broker at {BROKER}:{PORT} ...")
client.connect(BROKER, PORT, 60)
print("Connected ✅")

while True:
    # Simulated signal (temperature example)
    value = round(20 + random.uniform(-1, 1), 2)
    payload = {
        "sensor_id": SENSOR_ID,
        "value": value,
        "unit": "C"
    }
    client.publish(TOPIC, json.dumps(payload))
    print(f"📤 Sent: {payload}")
    time.sleep(3)  # send every 3 seconds


# docker build -t edge-sender .

# docker run --rm \
#   -e MQTT_HOST=192.168.68.251 \
#   -e MQTT_PORT=1883 \
#   -e MQTT_USER=server \
#   -e MQTT_PW=123456789 \
#   -e SENSOR_ID=edge1 \
#   -e MQTT_TOPIC='sensors/temperature' \
#   edge-sender

