import os
import paho.mqtt.client as mqtt

BROKER = os.getenv("MQTT_HOST", "mqtt-broker")
PORT   = int(os.getenv("MQTT_PORT", "1883"))
TOPIC  = os.getenv("MQTT_TOPIC", "sensors/#")

# Callback when connecting
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected to broker")
        client.subscribe(TOPIC)
        print(f"📡 Subscribed to topic: {TOPIC}")
    else:
        print(f"❌ Connection failed with code {rc}")

# Callback when a message arrives
def on_message(client, userdata, msg):
    print(f"📥 Topic: {msg.topic} | Message: {msg.payload.decode('utf-8')}")

# Initialize client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print(f"Connecting to {BROKER}:{PORT} ...")
client.connect(BROKER, PORT, 60)

client.loop_forever()
