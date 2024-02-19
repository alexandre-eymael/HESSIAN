from werkzeug.utils import secure_filename
import os
import base64

def parse_uploaded_image(app, query):
    """
    Parse the uploaded image and return it as a base64 string
    """
    if query and query.filename:
        filename = secure_filename(query.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        query.save(filepath)

        with open(filepath, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        image_data = f"data:image/jpeg;base64,{encoded_string}"
        return image_data