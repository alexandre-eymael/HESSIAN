# Stop existing container
echo "[1/3] Stopping existing container..."
sudo docker stop $(sudo docker ps -a -q)

# Build new image
echo "[2/3] Building new image..."
bash build_docker.sh

# Start new container
echo "[3/3] Starting new container..."
bash start_dockerized_server.sh

echo "DEPLOYMENT COMPLETED!"