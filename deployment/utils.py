"""
Utility Functions for Image Parsing and Processing

This module provides utility functions for parsing and processing uploaded images.
It includes a function to parse uploaded images, encode them into base64 strings,
and secure filenames.

Functions:
- parse_uploaded_image(app, query): Parses an uploaded image, encodes it into
  a base64 string, and secures the filename.

"""

import os
import base64
from werkzeug.utils import secure_filename

def parse_uploaded_image(app, query):
    """
    Parse the uploaded image and encode it into a base64 string.

    Args:
        app (Flask): The Flask application instance.
        query (FileStorage): The uploaded file.

    Returns:
        tuple: A tuple containing the encoded image as a bytes object
        and the string representation of the image.

    Raises:
        FileNotFoundError: If the uploaded file cannot be found.
        IOError: If an I/O error occurs during file processing.
    """
    if query and query.filename:
        filename = secure_filename(query.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        query.save(filepath)

        with open(filepath, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        str_image = f"data:image/jpeg;base64,{encoded_string.decode('utf-8')}"

        return encoded_string, str_image

    return None, None
