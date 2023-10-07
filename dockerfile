FROM python:3.11.5-bookworm
WORKDIR /root

COPY ["./installdeps.sh", "./requirements.txt","/root/"]

# Install any needed packages specified in requirements.txt
RUN /root/installdeps.sh
RUN pip3 install -r requirements.txt
RUN pip install gunicorn
EXPOSE 5000

# Copy the current directory contents into the container at /app
COPY . /root

RUN chmod -R 777 /root

# Define the command to run your app.py script

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app"]
# CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app.app:app"]
# CMD ["python", "app.py"]

