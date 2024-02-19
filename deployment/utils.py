from werkzeug.utils import secure_filename
import os
import base64

def parse_uploaded_image(app, query):
    if query and query.filename:
        filename = secure_filename(query.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        query.save(filepath)

        with open(filepath, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        str_image = f"data:image/jpeg;base64,{encoded_string.decode('utf-8')}"
        
        return encoded_string, str_image