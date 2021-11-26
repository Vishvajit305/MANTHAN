import tkinter as tk 
from tkinter import messagebox, ttk
from tkinter import *
import imutils
from imutils import paths
import numpy as np
import argparse
import pickle
from PIL import Image, ImageTk
import cv2
import os
import time
import pyautogui
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from threading import *
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json
from imutils.video import VideoStream
import time
import playsound


def recognize_video():
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
    # loop over frames from the video file stream
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
                if fW < 5 or fH < 5:
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
                        playsound.playsound(r'D:\Jithu Pgm working\Jithu\Vishva_programs\Ping Pong\wallhit.wav',False)
                elif(name.split('_')[0].isnumeric()):
                    cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
                    cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    if(check_spoof(face)=='spoof'):
                        cv2.putText(frame, check_spoof(face), (startX, endY+20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, check_spoof(face), (startX, endY+20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    pass
                    #cv2.imwrite((r"D:\Jithu Pgm working\Jithu\Vishva_programs\OPENCV_PLAY\face-recognition-using-deep-learning-master\face-recognition-using-deep-learning-master\dataset\Unknown\UNKNOWN.jpg"),frame)
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()
    return


def train_model():
    import Train_model
    """
    print("Loading Face Detector...")
    protoPath = "face_detection_model/deploy.prototxt"
    modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load serialized face embedding model
    print("Loading Face Recognizer...")
    embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

    # grab the paths to the input images in our dataset
    print("Quantifying Faces...")
    imagePaths = list(paths.list_images("dataset"))

    # initialize our lists of extracted facial embeddings and corresponding people names
    knownEmbeddings = []
    knownNames = []

    # initialize the total number of faces processed
    total = 0
    # loop over the image paths
    
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("Processing image {}/{}".format(i, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        # load the image, resize it to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # ensure at least one face was found
        if len(detections) > 0:
            # we're making the assumption that each image has only ONE face, so find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            # ensure that the detection with the largest probability also means our minimum probability test (thus helping filter out weak detections)
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI and grab the ROI dimensions
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # add the name of the person + corresponding face embedding to their respective lists
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1

    # dump the facial embeddings + names to disk
    print("[INFO] serializing {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    time.sleep(4)
    f = open("output/embeddings.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()

    print("[INFO] loading face embeddings...")
    data = pickle.loads(open("output/embeddings.pickle", "rb").read())

    # encode the labels
    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    # train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    print("[INFO] training model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    # write the actual face recognition model to disk
    f = open("output/recognizer.pickle", "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    f = open("output/le.pickle", "wb")
    f.write(pickle.dumps(le))
    f.close() 
    messagebox.showinfo("Result","Training is Done")
    return
    """

def create_dataset():
    def on_click():
        x=e0.get()
        y=e1.get()
        if(x.lower()=='yes' and y.lower()=='video'):
            import Terrorist_video
        elif(x.lower()=='yes' and y.lower()=='live'):
            import Terrorist_live
        elif(x.lower()=='no' and y.lower()=='video'):
            import Batchmate_video
        elif(x.lower()=='no' and y.lower()=='live'):
            import Batchmate_live
    root=tk.Tk()
    root.title("Entry point")
    e0=Entry(root,width=25)
    e0.pack(pady=15)
    e0.insert(0,"Is he a terrorist? If yes enter yes else enter no ")

    e1=Entry(root,width=25)
    e1.pack(pady=15)
    e1.insert(0,"Do you have the data as live or video")

    buttonx=Button(root,text="submit",command=on_click)
    buttonx.pack(pady=15)
    buttonx.focus
    root.geometry("500x500")

windows=tk.Tk()
windows.title("Face Detection System")

button0=tk.Button(windows,text="Create Dataset",font=("Algerian",17),bg="white",fg="black", command=create_dataset)
button0.pack(pady=30,padx=10)
#button0.grid(column=0,row=0)


button1=tk.Button(windows,text="Train Data",font=("Algerian",17),bg="white",fg="black",command=train_model)
button1.pack(pady=30)
#button1.grid(column=0,row=1)

button2=tk.Button(windows,text="Recognize",font=("Algerian",17),bg="white",fg="black", command=recognize_video)
button2.pack(pady=15)
#button2.grid(column=0,row=2)


windows.geometry("500x500")
windows.mainloop()