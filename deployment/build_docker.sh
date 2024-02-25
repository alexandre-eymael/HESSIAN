cd ..
sudo docker image build --progress=plain -t hessian_docker -f deployment/Dockerfile .
docker system prune -f