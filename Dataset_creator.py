import cv2
import numpy as np
import pyautogui
import os
from tkinter import * 

face_cascade = cv2.CascadeClassifier(r'D:\Jithu Pgm working\Jithu\Vishva_programs\OPENCV_PLAY\XML\haarcascade_frontalface_default.xml')

presenting=int(input("Are you for live or video?! \nIf video press 1 \nelse press 2 for live: "))
if(presenting==1):
	enter_the_path=input("Enter the path of the videofile: ").strip()
	cap = cv2.VideoCapture(enter_the_path)
else:	
	cap = cv2.VideoCapture(0)

name=input("Enter the name:").strip()
confirmation=input("Is he a terrorist?!\nEnter yes or no:")

while(1):
	if(confirmation.lower()=='yes'):
		terrorist=True
		break
	elif(confirmation.lower()=='no'):
		terrorist=False
		break

if(terrorist):
	flag=1
	while(flag):
		newpath = r'D:\Jithu Pgm working\Jithu\Vishva_programs\OPENCV_PLAY\face-recognition-using-deep-learning-master\face-recognition-using-deep-learning-master\dataset\Terrorist' 
		newpath+=f"\Terrorist_{name}"		
		if not os.path.exists(newpath):
			os.makedirs(newpath)
			flag=0
		else:
			name=input("Enter the name properly as a name already exisiting like that: ").strip()
	print(newpath)

else:
	identity=input("Enter the id number:").strip()
	flag=1
	while(flag):
		newpath = r'D:\Jithu Pgm working\Jithu\Vishva_programs\OPENCV_PLAY\face-recognition-using-deep-learning-master\face-recognition-using-deep-learning-master\dataset\Our Batch'
		newpath+=f"\{identity}_{name}"
		if not os.path.exists(newpath):
			os.makedirs(newpath)
			flag=0
		else:
			name=input("Enter the name properly as a name already exisiting like that: ").strip()
			identity=input("Enter the id number:").strip()
	print(newpath)


k=0
while(cap.isOpened()):
	return_value, img =cap.read()
	val=str(k).zfill(5)
	try:
		k+=1	
		cv2.imshow('Face_eye',img)
		cv2.imwrite(f"{newpath}/{val}.jpg",img)
		print(f"{newpath}/{val}.jpg")
		if (cv2.waitKey(1) & 0xFF == ord('q')) or (return_value==False):
			break  
	except:
		break
cap.release()
cv2.destroyAllWindows()    

#D:\Jithu Pgm working\Jithu\Vishva_programs\OPENCV_PLAY\face-recognition-using-deep-learning-master\face-recognition-using-deep-learning-master\videoplayback.mp4



"""def on_click():
	x=e0.get()
	yield x
	return x
	print(x)
root=Tk()
root.geometry("1000x200")

e0=Entry(root,width=25)
e0.pack()
e0.insert(0,"Enter the username: ")

button0=Button(root,text="submit",command=on_click)
button0.pack()
button0.focus

print(e0)
root.mainloop()"""