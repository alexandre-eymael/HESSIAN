import requests
import base64
from collections import OrderedDict
from pprint import pprint

URL = "http://hessian.be/api"

def _parse_image(image_path):
    """
    Parse the image as a path (either local or remote) and return the image encoded in base64.

    Args:
        image (str): The image to parse as a path (either local or remote).

    """
    try:
        with open(image_path, "rb") as f:
            image = f.read()
    except FileNotFoundError:
        try:
            response = requests.get(image_path)
            image = response.content
        except Exception as e:
            raise ValueError(f"Image {e} does not exist locally or remotely")
    return base64.b64encode(image).decode("utf-8")

def _top_k_predictions(predictions, k=5):
    """
    Get the top k predictions.

    Args:
        predictions (list): The predictions.
        k (int): The number of top predictions to return.

    Returns:
        list: The top k predictions.
    """
    return OrderedDict(sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:k])

def query_hessian_api(api_key, image, model_name):
    """
    Query the Hessian API using the key `api_key` and the image `image` with the model `model_name`.

    Args:
        api_key (str): The API key.
        image (str): The image to query as a path (either local or remote).
        model_name (str): The model name to query.
    """

    response = requests.post(
        URL,
        headers={"HESSIAN-API-Key": api_key},
        params={"model": model_name},
        data=_parse_image(image)
    )
    return response.json()

def query_billing(api_key):
    """
    Query the billing details using the key `api_key`.

    Args:
        api_key (str): The API key.

    Returns:
        dict: The billing details.
    """
    response = requests.get(
        f"{URL}/billing",
        headers={"HESSIAN-API-Key": api_key}
    )
    return response.json()

# Example usage
if __name__ == "__main__":

    # Your API key
    api_key = "d8c7cbcb-447d-4273-b285-5c88626b23be"

    ## Example #1 - Predictions

    # Cassava Mosaic Disease
    image = "https://agrio.app/wp-content/uploads/2019/11/image1-17.jpg"
    model_size = "large"

    # Query the API
    response = query_hessian_api(api_key, image, model_size)
    top_k = _top_k_predictions(response)
    
    print("<== Predictions ==>")
    pprint(top_k)

    print("\n")
    ## Example #2 - Billing Details
    
    # Query the billing details
    print("<== Billing Details ==>")
    billing = query_billing(api_key)
    pprint(billing)