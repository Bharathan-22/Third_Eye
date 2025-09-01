from  PIL import  Image
import pytesseract

tessdata_dir_config = '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
img_path='test.png'
text=pytesseract.image_to_string(Image.open(img_path), config=tessdata_dir_config)
 
print(text)
