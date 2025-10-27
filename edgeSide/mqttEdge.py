import os
import time
import json
import random
import paho.mqtt.client as mqtt

# Environment variables (set when running container)
BROKER      = os.getenv("MQTT_HOST", "192.168.68.251")
PORT        = int(os.getenv("MQTT_PORT", "1883"))
TOPIC       = os.getenv("MQTT_TOPIC", "sensors/temperature")
SENSOR_ID   = os.getenv("SENSOR_ID", "edge1")

client = mqtt.Client()

print(f"Connecting to MQTT broker at {BROKER}:{PORT} ...")
client.connect(BROKER, PORT, 60)
print("Connected ✅")

while True:
    # Simulated signal (for example, a temperature value)
    value = round(20 + random.uniform(-1, 1), 2)
    payload = {
        "sensor_id": SENSOR_ID,
        "value": value,
        "unit": "C"
    }
    client.publish(TOPIC, json.dumps(payload))
    print(f"📤 Sent: {payload}")
    time.sleep(3)  # send every 5 seconds
