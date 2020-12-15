from flask import Flask
import cv2
from test import MaskRecognizer
app = Flask(__name__)

@app.route("/")
def home():
	maskRecognizer = MaskRecognizer()
	return ("Is Elon wearing a mask? " + str(maskRecognizer.isWearingMask(cv2.imread('Tesla-Pickup.jpg'))))
    #return "Hello, World!"
    
if __name__ == "__main__":
    app.run(debug=True)