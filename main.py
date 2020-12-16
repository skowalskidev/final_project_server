from flask import Flask, request
from PIL import Image
import io
import base64
import cv2
import numpy as np
from test import MaskRecognizer
app = Flask(__name__)

maskRecognizer = MaskRecognizer()

@app.route("/")
def home():
	#maskRecognizer = MaskRecognizer()
	return ("ehm.Fukadahere")
	#return ("Is Elon wearing a mask? " + str(maskRecognizer.isWearingMask(cv2.imread('Tesla-Pickup.jpg'))))
	#return "Hello, World!"

@app.route("/isWearingMask", methods=["POST"])
def process_image():
	#file = request.files['image']
	# Read the image via file.stream
	#img = Image.open(file.stream)
	#img = Image.open(io.BytesIO(file.read()))
	#return "Hello"
	payload = request.form.to_dict(flat=False)
	im_b64 = payload['image'][0]  # remember that now each key corresponds to list.
	# see https://jdhao.github.io/2020/03/17/base64_opencv_pil_image_conversion/
	# for more info on how to convert base64 image to PIL Image object.
	im_binary = base64.b64decode(im_b64)
	buf = io.BytesIO(im_binary)
	img = Image.open(buf)
	#return "Got it"
	return maskRecognizer.isWearingMask((cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)))

if __name__ == "__main__": app.run(debug=True)