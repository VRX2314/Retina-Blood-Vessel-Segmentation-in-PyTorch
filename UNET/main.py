import gradio as gr
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from torchvision import transforms
from model import build_unet
from io import BytesIO
from PIL import Image


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)  ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask


def segment_image(input_image, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet()  # Replace with your model-building logic
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image = cv2.resize(input_image, (512, 512))
    x = np.transpose(image, (2, 0, 1))
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)

    with torch.no_grad():
        pred_y = model(x)
        pred_y = torch.sigmoid(pred_y)

        pred_y = pred_y[0].cpu().numpy()
        pred_y = np.squeeze(pred_y, axis=0)
        pred_y = pred_y > 0.5
        pred_y = np.array(pred_y, dtype=np.uint8)

        pred_y = mask_parse(pred_y)

    fig, ax = plt.subplots(1, 2)
    var = fig.set
    fig.size_inches(16, 9)
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[1].imshow(pred_y * 255)

    img_stream = BytesIO()
    plt.savefig(img_stream, format="png")
    img_stream.seek(0)

    plt.close()

    return Image.open(img_stream)


model_paths = [
    "../models/unet_smol_500.pth",
    "model_path_2.pth",
    "model_path_3.pth",
]  # Replace with your model paths

model_dropdown = gr.inputs.Dropdown(choices=model_paths, label="Select Model Path")
input_image = gr.inputs.Image(type="pil", label="Upload an Image")
output_image = gr.outputs.Image(type="pil", label="Segmented Image")

iface = gr.Interface(
    fn=segment_image,
    inputs=[model_dropdown, input_image],
    outputs=output_image,
    live=True,
)

if __name__ == "__main__":
    iface.launch()
