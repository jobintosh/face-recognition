FROM python:3.11.5-bookworm
WORKDIR /root

COPY ["./installdeps.sh", "./requirements.txt","/root/"]

RUN /root/installdeps.sh
RUN pip3 install -r requirements.txt
RUN pip install gunicorn
EXPOSE 5000

COPY . /root

RUN chmod -R 777 /root

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--log-level", "debug", "app"]


