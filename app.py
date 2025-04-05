import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model("SharpeningModel_512_30Epochs.keras", compile=False)

# Preprocessing
IMG_SIZE = (512, 512)

def preprocess(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def deblur_image(input_img):
    input_tensor = preprocess(input_img)
    prediction = model.predict(input_tensor)[0]
    prediction = np.clip(prediction, 0, 1)
    prediction = (prediction * 255).astype(np.uint8)
    return Image.fromarray(prediction)

# Gradio interface
demo = gr.Interface(
    fn=deblur_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="ReviveAI - Deblur Image",
    description="Upload a blurry image and get a sharper version using AI."
)

if __name__ == "__main__":
    demo.launch()
