import sys
import json
import os

if __name__ == "__main__":

    # -- ARGS
    if len(sys.argv) < 2:
        print("Usage: python update_models.py <dict>")

    raw_dict = sys.argv[1]
    parsed_dict = json.loads(raw_dict)

    epoch = parsed_dict["epoch"]
    model_size = parsed_dict["model_size"]
    path = parsed_dict["path"]

    remote_path = f"{path}/{model_size}/model_epoch_{epoch}.pt"
    # -- END ARGS

    # Start the comparison
    os.system(f"python3 -m deployment_train.compare_chkpts --size {model_size} --remote_checkpoint {remote_path} --verbose")