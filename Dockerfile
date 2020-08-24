FROM python:3.7-slim
WORKDIR /EnhanceIt
COPY requirements.txt .
RUN pip  install -r requirements.txt
COPY . .
CMD python3 src/main.py src/dataset/T91/1/tt17.png





