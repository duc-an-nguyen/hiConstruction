import os
import paho.mqtt.client as mqtt

# -------------------------------
# Config environment variables
# -------------------------------
BROKER = os.getenv("MQTT_HOST", "mqtt-broker")
PORT   = int(os.getenv("MQTT_PORT", "1883"))
TOPIC  = os.getenv("MQTT_TOPIC", "sensorsAndHealth/#")
USER   = os.getenv("MQTT_USER")
PW     = os.getenv("MQTT_PW")

# ===============================
# === Added for PostgreSQL ==== #
# ===============================
import json
import psycopg2
from psycopg2.extras import Json

PGHOST = os.getenv("PGHOST", "db")
PGPORT = int(os.getenv("PGPORT", "5432"))
PGDB   = os.getenv("PGDATABASE", "telemetry")
PGUSER = os.getenv("PGUSER", "grafana")      # you can rename
PGPW   = os.getenv("PGPASSWORD", "grafana")  # you can rename

def pg_connect():
    conn = psycopg2.connect(
        host=PGHOST, port=PGPORT, dbname=PGDB, user=PGUSER, password=PGPW
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                id        BIGSERIAL PRIMARY KEY,
                ts        TIMESTAMPTZ NOT NULL DEFAULT now(),
                topic     TEXT NOT NULL,
                rul       DOUBLE PRECISION,
                reacHI    DOUBLE PRECISION,
                sepHI     DOUBLE PRECISION,
                stripHI   DOUBLE PRECISION,
                sysHI     DOUBLE PRECISION,
                sensors   JSONB
            );
        """)
    return conn

_pg_conn = pg_connect()

def insert_row(topic: str, payload: dict):
    # known fields
    rul     = payload.get("rul")
    reacHI  = payload.get("reacHI")
    sepHI   = payload.get("sepHI")
    stripHI = payload.get("stripHI")
    sysHI   = payload.get("sysHI")
    # keep everything else as sensors JSON (if not provided)
    sensors = payload.get("sensors")
    if sensors is None:
        core = {"rul","reacHI","sepHI","stripHI","sysHI","ts","schema","seq"}
        sensors = {k:v for k,v in payload.items() if k not in core}

    with _pg_conn.cursor() as cur:
        cur.execute("""
            INSERT INTO telemetry(topic, rul, reacHI, sepHI, stripHI, sysHI, sensors)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (topic, rul, reacHI, sepHI, stripHI, sysHI, Json(sensors)))
# ===============================
# ===== End PostgreSQL add ===== #
# ===============================

def on_connect(client, userdata, flags, reason_code, properties):
    print("Connected rc:", reason_code)
    if reason_code == 0:
        client.subscribe(TOPIC, qos=1)
        print(f"Subscribed to: {TOPIC}")
    else:
        print("Not authorized or other connect error")

def on_message(client, userdata, msg):
    print(f"{msg.topic} {msg.payload.decode()}")

    # ===== Added: parse JSON and write to PostgreSQL =====
    try:
        data = json.loads(msg.payload.decode())
    except Exception as e:
        print("Bad JSON:", e)
        return
    try:
        insert_row(msg.topic, data)
        print("Inserted into PostgreSQL.")
    except Exception as e:
        print("DB insert error:", e)
    # ===== End add =====


# -------------------------------
# Create the client
# -------------------------------
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2) 

# -------------------------------
# Create user and password
# -------------------------------
if USER and PW:
    client.username_pw_set(USER, PW)

# -------------------------------
# Connect and receive the messages
# -------------------------------
print(f"Connecting to {BROKER}:{PORT} ...")
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT, 60)
client.loop_forever()
