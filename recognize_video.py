# import libraries
import os
import cv2
import imutils
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json
from imutils.video import VideoStream
import time
import playsound


face_cascade = cv2.CascadeClassifier(r'D:\Jithu Pgm working\Jithu\Vishva_programs\OPENCV_PLAY\XML\haarcascade_frontalface_default.xml')
json_file = open(r'D:\Jithu Pgm working\Jithu\Vishva_programs\Face_Antispoofing_System\antispoofing_models\97_accuracy-model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(r'D:\Jithu Pgm working\Jithu\Vishva_programs\Face_Antispoofing_System\antispoofing_models\97_accuracy-model.h5')
print("Model loaded from disk for spoof detection")

def check_spoof(faces):
	face = faces
	resized_face = cv2.resize(face,(160,160))
	resized_face = resized_face.astype("float") / 255.0
	# resized_face = img_to_array(resized_face)
	resized_face = np.expand_dims(resized_face, axis=0)
	# pass the face ROI through the trained liveness detector
	# model to determine if the face is "real" or "fake"
	preds = model.predict(resized_face)[0]
	#print(preds)
	if preds> 0.5:
		label = 'spoof'
	else:
		label = 'real'
	return label

# load serialized face detector
print("Loading Face Detector...")
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load serialized face embedding model
print("Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
#recognizer=cv2.face.LBPHFaceRecognizer_create()
le = pickle.loads(open("output/le.pickle", "rb").read())


# initialize the video stream, then allow the camera sensor to warm up
vs = VideoStream(src=0).start()
time.sleep(2.0)

print("Starting Video Stream...")		
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()

	# resize the frame to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
	frame = imutils.resize(frame, width=700)
	
	(h, w) = frame.shape[:2]
	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
		confidence = detections[0, 0, i, 2]
			
		# filter out weak detections
		if confidence > 0.75:
			# compute the (x, y)-coordinates of the bounding box for the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]
			
			# draw the bounding box of the face along with the associated probability
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.putText(frame,"CCTV NO.1",(0,25),cv2.FONT_HERSHEY_SIMPLEX,0.70,(0,0,0),2)
			cv2.putText(frame,"Date: ",(0,50),cv2.FONT_HERSHEY_SIMPLEX,0.70,(0,0,0),2)
			cv2.putText(frame,time.strftime("%d/%m/%y"),(55,50),cv2.FONT_HERSHEY_SIMPLEX,0.70,(0,0,0),2)
			cv2.putText(frame,"Time: ",(0,73),cv2.FONT_HERSHEY_SIMPLEX,0.70,(0,0,0),2)
			cv2.putText(frame,time.strftime("%H:%M:%S"),(60,75),cv2.FONT_HERSHEY_SIMPLEX,0.70,(0,0,0),2)
			
			if(name.split('_')[0]=='Terrorist'):
				cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
				cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
				if(check_spoof(face)=='spoof'):
					cv2.putText(frame, check_spoof(face), (startX, endY+20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
				else:
					cv2.putText(frame, check_spoof(face), (startX, endY+20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
					#playsound.playsound(r'D:\Jithu Pgm working\Jithu\Vishva_programs\Ping Pong\wallhit.wav',False)
			elif(name.split('_')[0].isnumeric()):
				cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
				cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				if(check_spoof(face)=='spoof'):
					cv2.putText(frame, check_spoof(face), (startX, endY+20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
				else:
					cv2.putText(frame, check_spoof(face), (startX, endY+20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()