import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

class MaskRecognizer:
	
	def __init__(self):
		self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		self.model = load_model("mask_recog_ver2.h5")
			
	def isWearingMask(self, image):
		image_capture = image
		gray = cv2.cvtColor(image_capture, cv2.COLOR_BGR2GRAY)
		faces = self.faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(60, 60),flags=cv2.CASCADE_SCALE_IMAGE)
		faces_list=[]
		preds=[]
		for (x, y, w, h) in faces:
			face_frame = image_capture[y:y+h,x:x+w]
			face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
			face_frame = cv2.resize(face_frame, (224, 224))
			face_frame = img_to_array(face_frame)
			face_frame = np.expand_dims(face_frame, axis=0)
			face_frame =  preprocess_input(face_frame)
			faces_list.append(face_frame)
			if len(faces_list)>0:
				preds = self.model.predict(faces_list)
			for pred in preds:
				(mask, withoutMask) = pred
			label = "Mask" if mask > withoutMask else "No Mask"
			if label == "No Mask":
				cv2.destroyAllWindows()
				return False
		cv2.destroyAllWindows()
		return True
