import cv2
import argparse
import time
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), -1)
    return frameOpencvDnn,faceBoxes


parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

video=cv2.VideoCapture(0)
padding=20

ret,frame1=video.read()
ret,frame2=video.read()

while(1):
    hasFrame,frame=video.read()

    diff=cv2.absdiff(frame1,frame2)
    gray=cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    _ ,thresh=cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    dialated=cv2.dilate(thresh,None,iterations=3)
    contors , _=cv2.findContours(dialated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.putText(frame,f"CCTV NO. {0}",(0,25),cv2.FONT_HERSHEY_SIMPLEX,0.70,(0,0,0),2)
    cv2.putText(frame,"Date: ",(0,50),cv2.FONT_HERSHEY_SIMPLEX,0.70,(0,0,0),2)
    cv2.putText(frame,time.strftime("%d/%m/%y"),(55,50),cv2.FONT_HERSHEY_SIMPLEX,0.70,(0,0,0),2)
    cv2.putText(frame,"Time: ",(0,73),cv2.FONT_HERSHEY_SIMPLEX,0.70,(0,0,0),2)
    cv2.putText(frame,time.strftime("%H:%M:%S"),(60,75),cv2.FONT_HERSHEY_SIMPLEX,0.70,(0,0,0),2)
    for i in contors:
        (x,y,w,h)=cv2.boundingRect(i)
        if(cv2.contourArea(i))<1000:
            continue
        cv2.putText(frame1,"Movement Detected",(120,20),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),3)
    cv2.drawContours(frame1,contors,-1,(0,255,0),2)        
    cv2.imshow("Feed",frame1)
    frame1=frame2
    ret,frame2=video.read()    
    
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        
        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()