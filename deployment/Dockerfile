# Start from the official Python image
FROM python:3.10.13-bookworm

# Set the working directory in the container
WORKDIR /app

# for the cache to work, we need to copy the requirements file first
COPY requirements/api_requirements.txt /app/requirements/api_requirements.txt
RUN pip install -r requirements/api_requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Start the server
ENTRYPOINT [ "python" ]
CMD ["-m", "deployment.server"]