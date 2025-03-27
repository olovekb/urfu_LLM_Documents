# app/Dockerfile

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt .

RUN pip3 install -r requirements.txt


COPY ./src .
#COPY config.toml /app/.streamlit/

#EXPOSE 8501
#EXPOSE 8005

#HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

#ENTRYPOINT ["streamlit", "run", "/app/src/app/streamlit/main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]

##CMD [ "python", "/app/src/app/api/main.py"]
