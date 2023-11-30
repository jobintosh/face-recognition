#!/bin/bash

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone the GitHub repository
git clone https://github.com/jobintosh/face-recognition
cd face-recognition

# Run Docker Compose
docker-compose -f "docker-compose.yml" up -d --build
