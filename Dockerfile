FROM clamsproject/clams-python-opencv4:0.5.2

RUN pip install --upgrade pip
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
WORKDIR /
RUN apt-get -y update
RUN apt-get -y install git
RUN git clone https://github.com/baudm/parseq.git
WORKDIR /parseq
RUN pip install -r requirements.txt

COPY ./app.py /parseq/app.py
WORKDIR /parseq

CMD ["python", "/parseq/app.py"]
