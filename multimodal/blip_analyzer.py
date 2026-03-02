from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# 加载模型
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def analyze_image(image_path):
    raw_image = Image.open(image_path).convert('RGB')

    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)

    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption
