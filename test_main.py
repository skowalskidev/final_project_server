import cv2
from test import MaskRecognizer
from time import sleep

def main():
	maskRecognizer = MaskRecognizer()
	print ("Is Elon wearing a mask? " + str(maskRecognizer.isWearingMask(cv2.imread('Tesla-Pickup.jpg'))))

if __name__ == "__main__":
	main()
