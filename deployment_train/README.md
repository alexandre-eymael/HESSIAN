# Dockerfile Overview
The `Dockerfile` is designed for containerizing our training script available at <a href="https://github.com/alexandre-eymael/HESSIAN/blob/main/models/train.py"> models/train </a>. This setup facilitates an isolated environment for training our models, ensuring consistency across different computing environments.

# Usage
To initiate the training process for our model within a Docker container, you will need to adjust the `run_dockerized_train.sh` script. This script should be configured with the necessary parameters for the training script, such as model type, batch size, and other relevant arguments. Following this setup, execute the command below:
```
bash deploy.sh
```
This command triggers the `Dockerfile` to construct the Docker container, setting up an environment where the model training is executed seamlessly.