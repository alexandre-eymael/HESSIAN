# Build new image
echo "Building new image..."
bash build_docker.sh

# Start new container
echo "Starting new container..."
bash run_dockerized_train.sh

echo "DEPLOYMENT COMPLETED!"