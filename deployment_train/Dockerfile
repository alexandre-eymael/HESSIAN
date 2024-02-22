# Start from the official Python image
FROM python:3.10.13-bookworm

# Set the working directory in the container
WORKDIR /app

# for the cache to work, we need to copy the requirements file first
COPY requirements/train/requirements.txt /app/requirements/train/requirements.txt
RUN pip install -r requirements/train/requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

ENTRYPOINT [ "python", "-m", "models.train" ]
