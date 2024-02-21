# Start from the official Python image
FROM python:3.10.13-bookworm

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Set up a volume for persistent data storage
VOLUME /app/data

# Run pip install for all the requirements in /app/requirements/*.txt
RUN ./requirements/install_dependencies.sh train

ENTRYPOINT [ "python" ]
CMD ["-m", "models/train"]