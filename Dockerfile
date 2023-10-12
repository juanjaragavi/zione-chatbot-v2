# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/juanjaragavi/zione-chatbot-v2.git .

RUN pip install -r requirements.txt

EXPOSE 8503

HEALTHCHECK CMD curl --fail http://localhost:8503/_stcore/health

ENTRYPOINT ["streamlit", "run", "zione_shop_chatbot_v2.py", "--server.port=8503", "--server.address=0.0.0.0"]