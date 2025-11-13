FROM python:3.12-slim


RUN apt-get update \
 && apt-get install -y --no-install-recommends libpq5 gcc python3-dev libpq-dev \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir paho-mqtt psycopg2-binary
RUN pip install gputil
RUN pip install --upgrade psutil
RUN pip install setuptools


WORKDIR /app

COPY mqttServer1.py .
COPY utility/ /app/utility/

CMD ["python", "mqttServer1.py"]




