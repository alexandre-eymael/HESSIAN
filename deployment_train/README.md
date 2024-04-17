# 🏋️ Training locally and in the Cloud
The code we use to train our models can be found in [models/train.py](../models/train.py). This section details how we configured and adapted this script to make it runnable both locally with Docker, and in the cloud with Vertex AI.

## 🐋 Dockerized Training Script

We containerized our training environment in a Dockerfile. The following sections explain how we configured it.

### 🐋 Docker File
We created a **[Dockerfile](Dockerfile)** to build a Docker image containing all the necessary requirements for training our model. This image is constructed using the command in [build_docker.sh](build_docker.sh) , which executes a straightforward `docker image build`. Once the image is ready, we deploy it to run our training script located at [models/train.py](../models/train.py). This process is detailed in [run_dockerized_train](run_dockerized_train.sh).

### 👁️ Training Monitoring with Weights & Biases
We utilize the [`wandb docker-run`](https://docs.wandb.ai/ref/cli/wandb-docker-run) command to integrate the wandb monitoring tool during the training process. This special command wraps the traditional `docker-run` command and properly starts a wandb-monitored training from the Dockerfile. Refer to the [models/](../models) directory to know more about how we used wandb during training.

### 💾 Docker Volumes
Additionally, we employed **Docker volumes** to synchronize data and model weights, ensuring that weights are saved directly on the host machine. This is crucial because the Docker container—and its contents—are removed once training concludes. More details on this setup can be found in [run_dockerized_train](run_dockerized_train.sh). 

### 🙅‍♂️ Docker Ignore
To prevent unnecessary data from being included in the Docker build, we created a [.dockerignore](../.dockerignore). This file is configured to exclude the synchronized data and weights from the build process.

## ☁️ Training in the Cloud: Vertex AI
In addition to be able to run the training locally with Docker, we also support training in the cloud with [Google's Vertex AI service](https://cloud.google.com/blog/topics/developers-practitioners/pytorch-google-cloud-how-train-and-tune-pytorch-models-vertex-ai?hl=en).

### 🐋 Container Image
Although it is possible to use Vertex AI with a custom Dockerfile, we did not use this method as it implied to build and upload a Docker Image of ~20GB, which we found overkilled as we only have a few packages to install. Instead, we used a [pre-built PyTorch image](https://europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest) provided by Vertex AI, in which we installed our dependencies.

### 📜 Dependencies
To use Vertex AI, it is required to package the training application through the creation of a Python source distribution. To do so, we created a [setup.py](../setup.py) file to turn our training script into a Python package. After that, we create our source distribution with the [package.sh](../package.sh) script that packages our training application into a `hessian_trainer-0.1.tar.gz` file and **uploads it in our training bucket in Google Cloud Storage**. This package encapsulates all of our dependencies and will be installed inside our pre-built image before training starts.

### 💾 Dataset
Our dataset is downloaded at the start of training. The URL of the training set can be modified in the arguments, which makes it easy to train our models on different datasets.

### 🚀 Starting the Training
The [start_vertex.py](start_vertex.py) script starts the Vertex AI training. Specifically, it starts by creating a custom train image from the pre-built image and installs the `hessian_trainer` package inside it. Then, a `CustomPythonPackageTrainingJob` is created, and started with our training arguments (i.e., hyperparameters and hardware to use).

### 👁️ Monitoring the Training
Similarly to the local training, we use Weights and Biases to monitor our training in the cloud. Specifically, we use **[Google Cloud Secret Manager](https://cloud.google.com/security/products/secret-manager)** to store and read our wandb API key so that it is never explicitely used in the code, which would be a security breach.

### ⛳ Saving the Checkpoints
Once training is finished, we upload the weights of our models to a Google Cloud Bucket.

### 🤖 CICD: Automated Training
We use GitHub Actions' Workflows to start a training automatically. Specifically, if a commit message contains the string `!train-hessian!`, a new training session is automatically started. Refer to the [training.yml](../.github/workflows/training.yml) workflow file the implementation.