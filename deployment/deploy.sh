# Stop existing container
sudo docker stop $(sudo docker ps -a -q)

# Build new image
bash build_docker.sh

# Start new container
bash start_dockerized_server.sh