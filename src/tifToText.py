import pytesseract
from PIL import Image
from unidecode import unidecode


fname = '../val_new/83M/BEIJING_HOUSE_LLC/lunchspecials.tif'
temp = Image.open(fname)
text = str(unidecode(pytesseract.image_to_string(temp)))
print text
