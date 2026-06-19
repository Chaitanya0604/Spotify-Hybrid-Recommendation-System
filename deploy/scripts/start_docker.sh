#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region ap-south-1 | sudo docker login --username AWS --password-stdin 114354607243.dkr.ecr.ap-south-1.amazonaws.com

echo "Pulling Docker image..."
sudo docker pull 114354607243.dkr.ecr.ap-south-1.amazonaws.com/spotify_hybrid_recsys:latest

echo "Checking for existing container..."
if [ "$(sudo docker ps -q -f name=spotify_hybrid_recsys)" ]; then
    echo "Stopping existing container..."
    sudo docker stop spotify_hybrid_recsys
fi

if [ "$(sudo docker ps -aq -f name=spotify_hybrid_recsys)" ]; then
    echo "Removing existing container..."
    sudo docker rm spotify_hybrid_recsys
fi

echo "Starting new container..."
sudo docker run -d -p 80:8000 --name spotify_hybrid_recsys 114354607243.dkr.ecr.ap-south-1.amazonaws.com/spotify_hybrid_recsys:latest

echo "Container started successfully."