FROM python:3.8-slim
WORKDIR /main_app
COPY . /main_app
RUN apt-get update && \
    apt-get install -y netcat-openbsd
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
