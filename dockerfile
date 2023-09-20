# Use the official Python image as a parent image
FROM python:3.11.5-bookworm

RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip3 install -r requirements.txt

# Define the command to run your app.py script
CMD ["python3", "app.py"]
