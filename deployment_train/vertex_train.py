from google.cloud import aiplatform
from datetime import datetime

# submit training job to Vertex Training with 
# custom container using Vertex SDK

APP_NAME = "hessian_train"
PROJECT_ID = "hessian-419310"
BUCKET_URI = "gs://hessian-bucket"

# initialize the Vertex SDK for Python
aiplatform.init(project=PROJECT_ID, staging_bucket=BUCKET_URI)

# define variable names
CUSTOM_TRAIN_IMAGE_URI = f"gcr.io/{PROJECT_ID}/pytorch_train_{APP_NAME}"
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
JOB_NAME = f"{APP_NAME}-pytorch-cstm-cntr-{TIMESTAMP}"

# configure the job with container image spec
job = aiplatform.CustomContainerTrainingJob(
    display_name=f"{JOB_NAME}", container_uri=f"{CUSTOM_TRAIN_IMAGE_URI}"
)

# define training code arguments
training_args = [
    "model_size", "small",
    "save_freq", "1",
    "epochs", "1",
    "lr", "3e-4",
    "train_prop", "0.8",
    "device", "cuda",
    "img_size", "224",
    "batch_size", "64",
    "wandb_mode", "online",
    "optimizer", "AdamW",
    "seed", "42"
]

# submit the Custom Job to Vertex Training service
model = job.run(
    replica_count=1,
    machine_type="n1-standard-8",
    accelerator_type=None, #TODO: Change
    accelerator_count=1,
    args=training_args,
    sync=False,
)