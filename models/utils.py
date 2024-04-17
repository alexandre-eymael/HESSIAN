import pathlib
import os
import json

def access_gcloud_secret(secret_name, project_id=154330559738, version=1):
    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_name}/versions/{version}"

    response = client.access_secret_version(name=name)

    return response.payload.data.decode("UTF-8")

def parse_data_path(raw_data_path, kaggle_username=None, kaggle_key=None):
    """
    Parse a raw data path into a pathlib.Path object.

    The data path can be:
    - A local path (e.g., /path/to/data)
    - A URL (e.g., https://example.com/data)

    If the data path is a URL, the data is downloaded to a local directory.
    """

    # Local Path
    if not raw_data_path.startswith("http"):
        return pathlib.Path(raw_data_path)
    
    # URL
    if "www.kaggle.com" in raw_data_path:

        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
            f.write(json.dumps({"username":kaggle_username,"key":kaggle_key}))

        import kaggle

        # Remove the potential trailing /data
        raw_data_path = raw_data_path.replace("/data", "")

        # Extract the dataset user and name
        parts = raw_data_path.split("/")

        user = parts[-2]
        dataset = parts[-1]

        dataset_name = f"{user}/{dataset}"
        kaggle.api.dataset_download_files(dataset=dataset_name, path="data", unzip=True, quiet=False)

        return pathlib.Path("data")
    
    raise ValueError("Invalid data path")


def upload_model_gs(model_path, bucket_name="hessian"):

    from google.cloud import storage

    # Connect and fetch the bucket
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    # Upload model
    blob = bucket.blob(model_path)
    blob.upload_from_filename(model_path)

    return f"{bucket_name}/{model_path}"