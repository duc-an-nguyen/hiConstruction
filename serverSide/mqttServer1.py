import os
import json
import psycopg2
from psycopg2.extras import Json
import paho.mqtt.client as mqtt


# -------------------------------
# Config environment variables for
# MQTT
# -------------------------------
BROKER = os.getenv("MQTT_HOST", "mqtt-broker")
PORT   = int(os.getenv("MQTT_PORT", "1883"))
TOPIC  = os.getenv("MQTT_TOPIC", "sensors/#")
USER   = os.getenv("MQTT_USER")
PW     = os.getenv("MQTT_PW")

# -------------------------------
# Config environment variables for
# PostgreSQL
# -------------------------------

PGHOST = os.getenv("PGHOST", "postgresql")
PGPORT = int(os.getenv("PGPORT", "5432"))
PGDB   = os.getenv("PGDATABASE", "telemetry")
PGUSER = os.getenv("PGUSER", "server")      
PGPW   = os.getenv("PGPASSWORD", "123456789")  

# =================================
# PostgreSQL functions
# ---------------------------------
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
                time      DOUBLE PRECISION,
                rul       DOUBLE PRECISION,
                reacHI    DOUBLE PRECISION,
                sepHI     DOUBLE PRECISION,
                stripHI   DOUBLE PRECISION,
                sysHI     DOUBLE PRECISION,
                sensors   JSONB
            );
        """)
    return conn

def insert_row(pg_conn, topic: str, payload: dict):
    # known fields
    time    = payload.get("time")
    rul     = payload.get("rul")
    reacHI  = payload.get("reacHI")
    sepHI   = payload.get("sepHI")
    stripHI = payload.get("stripHI")
    sysHI   = payload.get("sysHI")
    # keep everything else as sensors JSON (if not provided)
    sensors = payload.get("sensors")
    if sensors is None:
        core = {"Time", "rul","reacHI","sepHI","stripHI","sysHI","ts","schema","seq"}
        sensors = {k:v for k,v in payload.items() if k not in core}

    with pg_conn.cursor() as cur:
        cur.execute("""
            INSERT INTO telemetry(time, topic, rul, reacHI, sepHI, stripHI, sysHI, sensors)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (time, topic, rul, reacHI, sepHI, stripHI, sysHI, Json(sensors)))


# =================================
# MQTT function
# ---------------------------------

def on_connect(client, userdata, flags, reason_code, properties):
    print("Connected rc:", reason_code)
    if reason_code == 0:
        client.subscribe(TOPIC, qos=1)
        print(f"Subscribed to: {TOPIC}")
    else:
        print("Not authorized or other connect error")

def on_message(client, userdata, msg):
    print(f"{msg.topic} {msg.payload.decode()}")
    print("----------------------------------------------------")

    # --------------------------
    # Extract the data from message
    # --------------------------
    try:
        data = json.loads(msg.payload.decode())
    except Exception as e:
        print("Bad JSON:", e)
        return
    # --------------------------
    # Write into the database
    # --------------------------
    conn = userdata["pg_conn"]
    try:
        insert_row(conn,msg.topic, data)
        print("Inserted into PostgreSQL.")
    except Exception as e:
        print("DB insert error:", e)


# =================================
# MAIN FUNCTION
# =================================

if __name__ == '__main__':
    # -------------------------------
    # Create the client
    # -------------------------------
    pg_conn     = pg_connect()
    client      = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, userdata={"pg_conn": pg_conn}) 
    
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
