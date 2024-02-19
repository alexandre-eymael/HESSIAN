from AlexNet import create_AlexNet
import torch
from PIL import Image
from torchvision import transforms
import base64
import io

MODEL_TYPE_TO_PATH = {
    "small": "models/checkpoints/small.pt",
    "base": "models/checkpoints/base.pt",
    "large": "models/checkpoints/large.pt"
}

INDEX_TO_CLASS = {
    0: 'Apple___alternaria_leaf_spot', 1: 'Apple___black_rot', 2: 'Apple___brown_spot', 
    3: 'Apple___gray_spot', 4: 'Apple___healthy', 5: 'Apple___rust', 
    6: 'Apple___scab', 7: 'Bell_pepper___bacterial_spot', 8: 'Bell_pepper___healthy', 
    9: 'Blueberry___healthy', 10: 'Cassava___bacterial_blight', 11: 'Cassava___brown_streak_disease', 
    12: 'Cassava___green_mottle', 13: 'Cassava___healthy', 14: 'Cassava___mosaic_disease', 
    15: 'Cherry___healthy', 16: 'Cherry___powdery_mildew', 17: 'Corn___common_rust', 
    18: 'Corn___gray_leaf_spot', 19: 'Corn___healthy', 20: 'Corn___northern_leaf_blight', 
    21: 'Grape___black_measles', 22: 'Grape___black_rot', 23: 'Grape___healthy', 
    24: 'Grape___isariopsis_leaf_spot', 25: 'Grape_leaf_blight', 26: 'Orange___citrus_greening', 
    27: 'Peach___bacterial_spot', 28: 'Peach___healthy', 29: 'Potato___bacterial_wilt',
    30: 'Potato___early_blight', 31: 'Potato___healthy', 32: 'Potato___late_blight', 
    33: 'Potato___nematode', 34: 'Potato___pests', 35: 'Potato___phytophthora', 
    36: 'Potato___virus', 37: 'Raspberry___healthy', 38: 'Rice___bacterial_blight', 
    39: 'Rice___blast', 40: 'Rice___brown_spot', 41: 'Rice___tungro', 
    42: 'Soybean___healthy', 43: 'Squash___powdery_mildew', 44: 'Strawberry___healthy', 
    45: 'Strawberry___leaf_scorch', 46: 'Sugercane___healthy', 47: 'Sugercane___mosaic', 
    48: 'Sugercane___red_rot', 49: 'Sugercane___rust', 50: 'Sugercane___yellow_leaf', 
    51: 'Tomato___bacterial_spot', 52: 'Tomato___early_blight', 53: 'Tomato___healthy', 
    54: 'Tomato___late_blight', 55: 'Tomato___leaf_curl', 56: 'Tomato___leaf_mold', 
    57: 'Tomato___mosaic_virus', 58: 'Tomato___septoria_leaf_spot', 
    59: 'Tomato___spider_mites', 60: 'Tomato___target_spot'
    }

def load_model(model_type, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = create_AlexNet(num_classes=61, model_size=model_type)
    model.load_model(path=MODEL_TYPE_TO_PATH[model_type])
    model.to(device)
    return model

def predict_image(model, image):
    
    if isinstance(image, bytes):
        image = base64.decodebytes(image)
        image = Image.open(io.BytesIO(image)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4557, 0.4969, 0.3778], std=[0.1991, 0.1820, 0.2096]),
    ])
    image = transform(image) 
    image = image.unsqueeze(0)
    image = image.to(next(model.parameters()).device)
    
    # forward pass
    model.eval()
    with torch.no_grad(): 
        output = model(image)
        output = torch.nn.functional.softmax(output, dim=1)
    output = {INDEX_TO_CLASS[i]: output[0][i].item() for i in range(len(output[0]))}
    return output
        
if __name__ == '__main__':
    # ! do a parser for the arguments --> better for the API (merge with args_train.py ???)
    model = load_model('small')
    # encode the image to base64
    image = open('/home/badei/Projects/HESSIAN/data/images/Apple___alternaria_leaf_spot/000413.jpg', 'rb').read()
    image = base64.b64encode(image)
    print(predict_image(model, image))
    