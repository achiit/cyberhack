
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

app = FastAPI()

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

@app.post("/predict/")
async def predict(image: UploadFile = File(...), true_label: str = Form(...)):
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        
        face = mtcnn(pil_image)
        if face is None:
            return JSONResponse(status_code=400, content={"message": "No face detected"})

        face = face.unsqueeze(0)
        face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
        face = face.to(DEVICE, dtype=torch.float32) / 255.0

        target_layers=[model.block8.branch1[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(0)]

        grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(face.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(), grayscale_cam, use_rgb=True)
        face_with_mask = cv2.addWeighted(face.squeeze(0).permute(1, 2, 0).cpu().detach().numpy().astype('uint8'), 1, visualization, 0.5, 0)

        with torch.no_grad():
            output = torch.sigmoid(model(face).squeeze(0))
            prediction = "real" if output.item() < 0.5 else "fake"
            
            real_prediction = 1 - output.item()
            fake_prediction = output.item()
            
            confidences = {
                'real': real_prediction,
                'fake': fake_prediction
            }

            # Determine final prediction based on confidence scores
            final_prediction = "real" if real_prediction > fake_prediction else "fake"
        
        return {
            'confidences': confidences,
            'true_label': true_label,
            'final_prediction': final_prediction,
            'face_with_mask': face_with_mask.tolist()  # Convert numpy array to list for JSON serialization
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
