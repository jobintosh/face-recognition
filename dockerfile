# Use the official Python image as a parent image
FROM python:3.11.5-bookworm

# RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Set the working directory to /app
WORKDIR /app

COPY ["./installdeps.sh", "./requirements.txt","/app/"]

# Install any needed packages specified in requirements.txt
RUN /app/installdeps.sh
RUN pip3 install -r requirements.txt
RUN pip install gunicorn
EXPOSE 5000

# Copy the current directory contents into the container at /app
COPY . /app

RUN chmod -R 777 /app

# Define the command to run your app.py script

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app.app"]
