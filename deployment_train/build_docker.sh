cd ..
docker image build --progress=plain -t hessian_docker_train -f deployment_train/Dockerfile .
docker system prune -f