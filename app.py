from flask import *
import requests
import pickle
import pytesseract
import re
import urllib.request
import speech_recognition as sr
from PIL import *
from io import BytesIO
from sightengine.client import SightengineClient
from googletrans import Translator
from urlextract import URLExtract

app = Flask(__name__)

hate_model = pickle.load(open(r"hatespeech\saved_models\lr_model.pkl", 'rb'))  
hate_vect = pickle.load(open(r"hatespeech\saved_models\vectorizer.pkl", 'rb')) 


custom_config = r'--oem 3 --psm 6'

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

client = SightengineClient('510188098','WaehbLBjT3mYTmnxDsp3')
extractor = URLExtract()


def ishate(string:str) -> bool:
    string=[string]
    sen_trans = hate_vect.transform(string)
    prediction = hate_model.predict(sen_trans)[0]  #0->normal 1->toxic
    return True if prediction else False

@app.route('/', methods=['POST','GET'])
def main():
    data = request.get_json()

    t = data["type"]
    translator = Translator()

    if t=="text":
        receivedtext = data["data"]
        receivedtext = translator.translate(receivedtext).text
        if ishate(receivedtext):
            return jsonify(False)
        return jsonify(True)


if __name__=="__main__":
    app.run(debug=True)
