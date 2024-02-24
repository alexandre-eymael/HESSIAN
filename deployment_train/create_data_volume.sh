cd ..
# create the volume 'data_volume' for the data container
docker volume create data_volume
docker run --rm -v data_volume:/app/data -v ./data:/source alpine cp -r /source/. /app/data
