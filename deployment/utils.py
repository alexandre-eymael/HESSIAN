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

        # Option 1: Pass the file path to the template (if serving the file through a static route)
        # image_url = url_for('static', filename='uploads/' + filename)

        # Option 2: Convert the image to base64 and pass it directly to the template
        with open(filepath, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        image_data = f"data:image/jpeg;base64,{encoded_string}"
        return image_data
    
def validate_api_key(db, api_key):
    return api_key in db