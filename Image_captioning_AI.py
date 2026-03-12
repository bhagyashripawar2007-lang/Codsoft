import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings
warnings.filterwarnings("ignore")
from transformers import BlipProcessor, BlipForConditionalGeneration

from PIL import Image
import torch

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20)

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def main():
    image_path = "dog.jpg"   # ensure same folder
    print("Generating caption for image...\n")
    result = generate_caption(image_path)
    print("AI Caption:", result)

if __name__ == "__main__":
    main()