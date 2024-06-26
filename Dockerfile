FROM python:3.11-slim 

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
RUN python -m playwright install
RUN python -m playwright install-deps

COPY . .

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run" , "main.py", "--server.port=8501","--server.address=0.0.0.0"]

