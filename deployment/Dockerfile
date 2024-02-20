# Start from the official Python image
FROM python:3.10.13-bookworm

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN ls -la /app

# Run pip install for all the requirements in /app/requirements/*.txt
RUN ./requirements/install_dependencies.sh api

# configure the container to run in an executed manner
ENTRYPOINT [ "python" ]

CMD ["-m", "flask", "--app", "deployment/server", "run", "--debug"]