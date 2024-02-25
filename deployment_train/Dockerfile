# Start from the official Python image
FROM python:3.10.13-bookworm

# Set the working directory in the container
WORKDIR /app

# for the cache to work, we need to copy the requirements file first
COPY requirements/train_requirements.txt /app/requirements/train_requirements.txt
RUN pip install -r requirements/train_requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

ENTRYPOINT [ "python", "-m", "models.train" ]
