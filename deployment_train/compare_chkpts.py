import torch
import os

def args():
    import argparse

    parser = argparse.ArgumentParser(description="Compare two model checkpoints.")
    parser.add_argument("--size", type=str, default="small", help="The size of the model.", choices=["small", "large", "base"])
    parser.add_argument("--remote_checkpoint", type=str, default="models/weights/small.pt", help="The path to the first checkpoint.")
    parser.add_argument("--tmp_location", type=str, default="/tmp/model.pt", help="The path to save the remote checkpoint.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")

    return parser.parse_args()

def compare_chkpts(path1, path2, metric="test_acc", verbose=False):
    """
    Compare two model checkpoints by reading the "metrics" key from the checkpoint.

    Args:
        path1 (str): The path to the first checkpoint.
        path2 (str): The path to the second checkpoint.

    Returns:
        path_i : The path to the checkpoint with the highest accuracy.
    """
    checkpoint1 = torch.load(path1)
    checkpoint2 = torch.load(path2)

    metric1 = checkpoint1["metrics"][metric]
    metric2 = checkpoint2["metrics"][metric]

    if verbose:
        print(f"Checkpoint 1 {path1} {metric} = {metric1}")
        print(f"Checkpoint 2 {path2} {metric} = {metric2}")

    if metric1 > metric2:
        return path1
    else:
        return path2
    
def fetch_model_gcp(gcp_path, local_path="/tmp/model.pt"):
    """
    Fetch a model from Google Cloud Storage.

    Args:
        gcp_path (str): The path to the model in Google Cloud Storage.

    Returns:
        None
    """
    from google.cloud import storage

    # Connect and fetch the bucket
    client = storage.Client()
    bucket = client.get_bucket("hessian")

    # Download the model
    blob = bucket.blob(gcp_path)
    blob.download_to_filename(local_path)

    return local_path

if __name__ == "__main__":

    args = args()

    if args.verbose:
        print(f"Downloading the remote checkpoint from {args.remote_checkpoint} and saving it to {args.tmp_location}...")

    remote_checkpoint = fetch_model_gcp(args.remote_checkpoint, local_path=args.tmp_location)
    local_checkpoint = f"models/checkpoints/{args.size}.pt"

    if args.verbose:
        print(f"Comparing the remote checkpoint {remote_checkpoint} with the local checkpoint {local_checkpoint}...")

    # Compare
    best_checkpoint = compare_chkpts(remote_checkpoint, local_checkpoint, verbose=args.verbose)
    
    if args.verbose:
        print(f"Moving the best checkpoint to {local_checkpoint}...")

    # Move the best checkpoint to the local directory
    os.rename(best_checkpoint, local_checkpoint)
        