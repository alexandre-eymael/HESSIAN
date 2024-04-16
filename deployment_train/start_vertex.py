from google.cloud import aiplatform
from datetime import datetime

# --- CONFIG
PRE_BUILT_TRAINING_CONTAINER_IMAGE_URI = "europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest"
PYTHON_PACKAGE_APPLICATION_DIR = "."

APP_NAME = "hessian"
PROJECT_ID = "hessian-419310"
BUCKET_URI = "gs://hessian"
LOCATION = "europe-west1"

source_package_file_name = f"{PYTHON_PACKAGE_APPLICATION_DIR}/dist/hessian_trainer-0.1.tar.gz"
python_package_gcs_uri = f"{BUCKET_URI}/pytorch-on-gcp/{APP_NAME}/train/python_package/hessian_trainer-0.1.tar.gz"

python_module_name = "models.train"
# ---

# initialize the Vertex SDK for Python
aiplatform.init(project=PROJECT_ID, staging_bucket=BUCKET_URI)

# define variable names
CUSTOM_TRAIN_IMAGE_URI = f"gcr.io/{PROJECT_ID}/pytorch_train_{APP_NAME}"
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
JOB_NAME = f"{APP_NAME}-pytorch-pkg-ar-{TIMESTAMP}"

print(f"APP_NAME={APP_NAME}")
print(f"PRE_BUILT_TRAINING_CONTAINER_IMAGE_URI={PRE_BUILT_TRAINING_CONTAINER_IMAGE_URI}")
print(f"python_package_gcs_uri={python_package_gcs_uri}")
print(f"python_module_name={python_module_name}")
print(f"JOB_NAME={JOB_NAME}")

# configure the job with container image spec
job = aiplatform.CustomPythonPackageTrainingJob(
    display_name=f"{JOB_NAME}",
    python_package_gcs_uri=python_package_gcs_uri,
    python_module_name=python_module_name,
    container_uri=PRE_BUILT_TRAINING_CONTAINER_IMAGE_URI,
    location=LOCATION
)

EPOCHS = 1
MODEL_SIZE = "small"
PATH = "weights"

# define training code arguments
training_args = [
    "--data_path", "https://www.kaggle.com/datasets/nirmalsankalana/plant-diseases-training-dataset/data",
    "--model_size", f"{MODEL_SIZE}",
    "--save_freq", "1",
    "--epochs", f"{EPOCHS}",
    "--lr", "3e-4",
    "--train_prop", "0.8",
    "--device", "cpu",
    "--img_size", "224",
    "--batch_size", "64",
    "--wandb_mode", "online",
    "--optimizer", "AdamW",
    "--save_path", f"{PATH}",
    "--seed", "42",
    "--vertex_ai" # add this flag to indicate that the training is running on Vertex AI to retrieve gcloud secrets
]

# Submit the Custom Job to Vertex Training service
job.run(
    replica_count=1,
    machine_type="n1-standard-8",
    #accelerator_type="NVIDIA_TESLA_P100",
    #accelerator_count=1,
    args=training_args,
    sync=True,
    service_account="hessian@hessian-419310.iam.gserviceaccount.com"
)

# Output
import json
output_dict = {"epoch": EPOCHS, "model_size": MODEL_SIZE, "path": PATH, "job_name": JOB_NAME}
print(json.dumps(output_dict))