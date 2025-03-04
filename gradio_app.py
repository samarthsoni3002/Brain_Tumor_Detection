import torch 
from save_and_load_model_weights import load_model
from data_preparation import transformers
from PIL import Image
from utils import get_class_names
import gradio as gr

class_name,_,_ = get_class_names("./brain_tumor_dataset/Training")

model = load_model(model_path="brain_tumor_model.pth")
_,test_transforms = transformers()

def predict(image: Image.Image):
    image = test_transforms(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        prediction = model(image)
        prediction = torch.argmax(prediction)
        return class_name[prediction.item()]


iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Brain Tumor Detection",
    description="Upload an MRI scan to detect whether a brain tumor is present."
)

iface.launch()