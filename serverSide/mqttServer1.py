import os
import json
import psycopg2
from psycopg2.extras import Json
import paho.mqtt.client as mqtt
import psutil
import GPUtil
import time




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
                sensors   JSONB,
                cpu       DOUBLE PRECISION,
                mem       DOUBLE PRECISION,
                gpuLoad   DOUBLE PRECISION,
                gpuTemp   DOUBLE PRECISION
            );
        """)
    return conn

def get_gpu_stats():
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


def insert_row(pg_conn, topic: str, payload: dict):
    # known fields
    time    = payload.get("time")
    rul     = payload.get("rul")
    reacHI  = payload.get("reacHI")
    sepHI   = payload.get("sepHI")
    stripHI = payload.get("stripHI")
    sysHI   = payload.get("sysHI")
    latencyTime = payload.get("latencyTime")

    cpu     = psutil.cpu_percent(interval=1)
    mem     = psutil.virtual_memory().percent
    gpuLoad, gpuTemp = get_gpu_stats()


    # keep everything else as sensors JSON (if not provided)
    sensors = payload.get("sensors")
    if sensors is None:
        core = {"Time", "rul","reacHI","sepHI","stripHI","sysHI","ts","schema","seq"}
        sensors = {k:v for k,v in payload.items() if k not in core}

    with pg_conn.cursor() as cur:
        cur.execute("""
            INSERT INTO telemetry(time, topic, rul, reacHI, sepHI, stripHI, sysHI, cpu, mem, gpuLoad, gpuTemp, sensors)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (time, topic, rul, reacHI, sepHI, stripHI, sysHI, cpu, mem, gpuLoad, gpuTemp, Json(sensors)))

    print(f"The latency time is:{(time.time() - latencyTime)*2}")
# =================================
# MQTT function
# ---------------------------------

def on_connect(client, userdata, flags, reason_code, properties):
    print("Connected rc:", reason_code)
    if reason_code == 0:
        client.subscribe(TOPIC, qos=0)
        print(f"Subscribed to: {TOPIC}")
    else:
        print("Not authorized or other connect error")



recv_count = 0
dup_count = 0
seen_seq = set()
total_payload_bytes = 0
latency_list = []
first_recv_time = None

def print_final_results():
    total_time      = time.time() - first_recv_time if first_recv_time else 1
    delivery_ratio  = 100 * len(seen_seq) / (max(seen_seq)+1)  # approximate total sent
    throughput      = len(seen_seq) / total_time
    goodput         = (8 * total_payload_bytes) / (total_time * 1e6)  # Mb/s

    print("\n========== FINAL RESULTS ==========")
    print(f"Total messages received: {recv_count}")
    print(f"Total duplicates: {dup_count}")
    print(f"Delivery ratio: {delivery_ratio:.2f}%")
    print(f"Throughput: {throughput:.3f} msg/s")
    if latency_list:
        print(f"Average latency: {sum(latency_list)/len(latency_list):.2f} ms")
        print(f"p95 latency: {sorted(latency_list)[int(0.95*len(latency_list))-1]:.2f} ms")
    print(f"Goodput: {goodput:.4f} Mb/s")
    print("===================================")

def on_message(client, userdata, msg):
    print(f"{msg.topic} {msg.payload.decode()}")
    print("----------------------------------------------------")
    t_recv = time.time()
    # --------------------------
    # Extract the data from message
    # --------------------------
    try:
        data = json.loads(msg.payload.decode())
    except Exception as e:
        print("Bad JSON:", e)
        return

    # --------------------------
    # Extract publish timestamp and sequence number
    seq     = data.get("seq")
    t_pub   = data.get("pub_ts")

    if seq in seen_seq:
        dup_count += 1
        print(f"[DUPLICATE] seq={seq}")
        return
    else:
        seen_seq.add(seq)

    recv_count += 1
    payload_bytes = len(msg.payload)
    total_payload_bytes += payload_bytes

    # Measure latency if t_pub exists
    if t_pub is not None:
        latency = (t_recv - t_pub) * 1000.0  # convert to ms
        latency_list.append(latency)
        print(f"[LATENCY] seq={seq} latency={latency:.2f} ms")




    # --------------------------
    # Write into the database
    # --------------------------
    conn = userdata["pg_conn"]
    try:
        insert_row(conn,msg.topic, data)
        print("Inserted into PostgreSQL.")
    except Exception as e:
        print("DB insert error:", e)

    
    # For first message, set start time for throughput calc
    if first_recv_time is None:
        first_recv_time = t_recv


    if seq==500:
        print_final_results()

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
