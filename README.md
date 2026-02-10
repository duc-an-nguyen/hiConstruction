# Edge–Server AI System

This repository implements an edge–server architecture for AI processing, data communication, and visualization.  
The system is divided into two main parts:

- **Server Side**: Centralized services, data storage, and monitoring
- **Edge Side**: Data acquisition and AI inference on the edge device

---

## System Architecture Overview

- The **edge device** runs AI models locally and sends data to the server via MQTT.
- The **server** manages message handling, database storage, and visualization using Grafana.
- All services on the server are containerized using Docker.

---

## Server Side

### Location

```
serverSide/
```

### Components

- AI models
- MQTT services
- PostgreSQL database
- Grafana dashboards

### Key Files

- **Main service entry point**

```
aiModel/mqttServer1.py
```

This script initializes and connects:
- MQTT broker
- MQTT receiver
- PostgreSQL database
- Grafana integration

- **Service orchestration**

```
docker-compose.yml
```

---

### How to Run the Server

#### Prerequisites

- Docker (Ubuntu or Windows)

Install Docker using the official documentation:  
https://docs.docker.com/engine/install/

#### Start Services

```bash
cd yourDirectory/serverSide
docker compose up --build
```

#### Grafana Access

After the services start successfully, open:

```
http://localhost:3000/
```

---

## Edge Side

### Location

```
edgeSide/
```

### Main Files

```
main.py                 # Application entry point
input_signal2_ai.py     # Input data acquisition (multi-threaded)
ai_handling.py          # AI inference and processing
edge_communication.py  # MQTT communication with the server
```

---

### How to Run the Edge Application

#### Python Environment

- A Python environment must be available on the edge device.
- The environment is located in the `env` folder on the **Jetson Nano**.

#### Network Setup

- The edge device and server must be on the same network.

Find the server IP address:

```
Windows: ipconfig
Linux/macOS: ifconfig or ip a
```

Update the MQTT broker address in `edge_communication.py`:

```python
BROKER = os.getenv("MQTT_HOST", "10.235.117.251")
```

Replace the IP address with the actual server IP.

#### Start the Edge Application

```bash
python main.py
```

---

## Usage Notes

- Start the **server side first** before launching the edge application.
- Wait until all Docker containers are fully running.
- Ensure stable network connectivity between edge and server.
- Verify MQTT communication if data does not appear in Grafana.

---

## Supported Platforms

- **Server**: Ubuntu / Windows (Docker-based)
- **Edge**: NVIDIA Jetson Nano
