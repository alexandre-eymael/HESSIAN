# Start from the official Python image
FROM python:3.10.13-bookworm

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Run pip install for all the requirements in /app/requirements/*.txt
RUN ./requirements/install_dependencies.sh api

# Start the server
ENTRYPOINT [ "python" ]
CMD ["-m", "flask", "--app", "deployment/server", "run", "--host=0.0.0.0", "--port=80"]