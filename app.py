from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
import io
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from pathlib import Path
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Define the classes
CLASSES = ['healthy', 'infected']

# Define transforms for data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the model
class ImageClassifier(torch.nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, (3,3)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, (3,3)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, (3,3)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128 * 26 * 26, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, len(CLASSES))
        )

    def forward(self, x):
        return self.model(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

try:
    model = ImageClassifier().to(device)
    model.load_state_dict(torch.load('models/PlantDiseaseAlgo.pt', map_location=device))
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Initialize validation model (MobileNetV3)
val_model = None
try:
    val_weights = MobileNet_V3_Small_Weights.DEFAULT
    val_model = mobilenet_v3_small(weights=val_weights).to(device)
    val_model.eval()
    logger.info("Validation model loaded successfully")
except Exception as e:
    logger.warning(f"Error loading validation model: {e}. Plant validation will be skipped.")

# Plant-related class indices in ImageNet-1K
PLANT_CATEGORY_INDICES = {
    580,  # greenhouse
    738,  # pot, flowerpot
    924,  # guacamole (avocado)
    935,  # mashed potato
    936,  # head cabbage
    937,  # broccoli
    938,  # cauliflower
    939,  # zucchini
    940,  # spaghetti squash
    941,  # acorn squash
    942,  # butternut squash
    943,  # cucumber
    944,  # artichoke
    945,  # bell pepper
    946,  # cardoon
    947,  # mushroom
    948,  # Granny Smith (apple)
    949,  # strawberry
    950,  # orange
    951,  # lemon
    952,  # fig
    953,  # pineapple
    954,  # banana
    955,  # jackfruit
    956,  # custard apple
    957,  # pomegranate
    958,  # hay
    984,  # rapeseed
    985,  # daisy
    986,  # yellow lady's slipper
    987,  # corn
    988,  # acorn
    989,  # hip, rose hip
    990,  # buckeye
    991,  # coral fungus
    992,  # agaric
    993,  # gyromitra
    994,  # stinkhorn
    995,  # earthstar
    996,  # hen-of-the-woods
    997,  # bolete
    998,  # ear, spike, capitulum
}

def is_valid_plant(image_bytes):
    """
    Validates whether the image contains a plant/leaf/fruit/vegetable.
    Returns True if valid, False otherwise.
    """
    if val_model is None:
        return True
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = val_model(img_tensor)
            _, top5_indices = torch.topk(outputs[0], 5)
            top5_set = set(top5_indices.tolist())
            # If any of the top 5 predicted categories are plant-related, accept
            is_plant = len(top5_set.intersection(PLANT_CATEGORY_INDICES)) > 0
            logger.info(f"Plant validation top-5 classes: {top5_indices.tolist()}, Is Plant: {is_plant}")
            return is_plant
    except Exception as e:
        logger.error(f"Error in plant validation: {e}")
        return True

def predict_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()
            
            logger.info(f"Prediction: {CLASSES[predicted.item()]}, Confidence: {confidence:.4f}")
            return CLASSES[predicted.item()], confidence
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    try:
        if 'image' not in request.files:
            logger.warning("No image in request")
            return jsonify({'error': 'No image uploaded'}), 400
        
        image_bytes = request.files['image'].read()
        
        if not image_bytes:
            logger.warning("Empty image")
            return jsonify({'error': 'Empty image'}), 400
            
        # Validate if the image is a plant
        if not is_valid_plant(image_bytes):
            logger.warning("Validation failed: Image is not a valid plant")
            return jsonify({'error': 'Incorrect input: The uploaded image does not appear to contain a valid plant or leaf.'}), 400
            
        prediction, confidence = predict_image(image_bytes)
        
        return jsonify({
            'prediction': prediction,
            'confidence': f"{confidence * 100:.2f}%"
        })
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)