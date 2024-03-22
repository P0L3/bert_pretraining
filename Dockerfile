FROM python:3.8.0

WORKDIR /bert_pretraining

COPY ./requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

RUN mkdir PRETRAINING

CMD ["echo", "PRETRAINING container ready!"]
