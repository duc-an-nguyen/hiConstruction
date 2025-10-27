# mqttServer.py
import os
import paho.mqtt.client as mqtt

BROKER = os.getenv("MQTT_HOST", "mqtt-broker")
PORT   = int(os.getenv("MQTT_PORT", "1883"))
TOPIC  = os.getenv("MQTT_TOPIC", "sensors/#")
USER   = os.getenv("MQTT_USER")
PW     = os.getenv("MQTT_PW")

def on_connect(client, userdata, flags, reason_code, properties):
    print("Connected rc:", reason_code)
    if reason_code == 0:
        client.subscribe(TOPIC, qos=1)
        print(f"Subscribed to: {TOPIC}")
    else:
        print("Not authorized or other connect error")

def on_message(client, userdata, msg):
    print(f"{msg.topic} {msg.payload.decode()}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)  # fixes deprecation
if USER and PW:
    client.username_pw_set(USER, PW)

print(f"Connecting to {BROKER}:{PORT} ...")
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT, 60)
client.loop_forever()
