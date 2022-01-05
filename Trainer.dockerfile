
   
FROM python:3.7-slim

RUN apt-get -y update
RUN apt-get -y upgrade

RUN apt-get -y install \
    make gcc g++ python3 python3-dev libc-dev

RUN apt-get -y install git

WORKDIR /

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/

RUN pip install -r requirements.txt --no-cache-dir

ENV DIRECTORY="/"

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]

# CMD ["sh", "-c", "tail -f /dev/null"]