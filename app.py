from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np

app = Flask(__name__)

# Device configuration
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load the face detection model
mtcnn = MTCNN(select_largest=False, post_process=False, device=DEVICE).eval()

# Load the face recognition model
model = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=1, device=DEVICE).eval()

def predict(input_image):
    """Predict the label of the input_image"""
    # Face detection
    face = mtcnn(input_image)
    if face is None:
        raise Exception('No face detected')
    
    # Preprocess the face image
    face = F.interpolate(face.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
    face = face.to(DEVICE, dtype=torch.float32) / 255.0
    
    # Forward pass through the model
    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        prediction = "real" if output.item() < 0.5 else "fake"
        
        real_prediction = 1 - output.item()
        fake_prediction = output.item()
        
        confidences = {
            'real': real_prediction,
            'fake': fake_prediction
        }
    return confidences, prediction

@app.route('/predict', methods=['POST'])
def handle_prediction():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        input_image = Image.open(file)
        confidences, prediction = predict(input_image)
        return jsonify({
            'confidences': confidences,
            'prediction': prediction
        })

if __name__ == '__main__':
    app.run(debug=True)
