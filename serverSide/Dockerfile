
FROM tensorflow/tensorflow:2.16.1-gpu


# --------------------------------------
# Packages for services
# --------------------------------------

RUN apt-get update \
 && apt-get install -y --no-install-recommends libpq5 gcc python3-dev libpq-dev \
 && rm -rf /var/lib/apt/lists/*


RUN python -m pip install --no-cache-dir paho-mqtt psycopg2-binary
RUN python -m pip install gputil
RUN python -m pip install --upgrade psutil
RUN python -m pip install setuptools

# --------------------------------------
# Package for AI
# --------------------------------------
    
RUN python -m pip install pandas==2.1.4
RUN python -m pip install scipy==1.10.1
RUN python -m pip install tables==3.9.2
RUN python -m pip install joblib
# RUN pip install --upgrade setuptools pip wheel
RUN python -m pip install matplotlib==3.8.2
RUN python -m pip install prettytable==3.9.0





WORKDIR /app

# Now /app is the default folder, copy into it 
# COPY mqttServer1.py .                 

CMD ["python", "mqttServer1.py"]
