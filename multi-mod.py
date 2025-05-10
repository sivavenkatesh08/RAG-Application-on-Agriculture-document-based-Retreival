import pytesseract
from PIL import Image
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def extract_info(image_path):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for better OCR accuracy
    extracted_text = pytesseract.image_to_string(gray_img)

    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return extracted_text, caption

image_path = "your-img-path"  # Replace with your image file path
text, caption = extract_info(image_path)

print("Extracted Text (OCR):")
print(text)
print("\nImage Caption (Description):")
print(caption)
