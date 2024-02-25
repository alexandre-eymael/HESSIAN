"""
Tests for the API
"""

import requests

URLS_API = ["http://hessian.be/api", "http://hessian.be/api/billing"]

def test_api_status():
    """
    Test that the API is:
        (1) reachable
        (2) available (i.e. returns status code 200)
    """

    for url in URLS_API:

        # (1) Check if the API is reachable
        try:
            response = requests.get(url, timeout=5)
        except requests.exceptions.RequestException as e:
            error = f"Error occured while trying to API `{url}`: {e}"
            raise requests.exceptions.RequestException(error)

        # (2) Check if the API is available
        assert response.status_code == 200, \
            f"API `{url}` is reachable but returned status code {response.status_code}"
