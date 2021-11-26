import cv2
import numpy as np
import pyautogui
import os
from tkinter import * 
from tkinter import messagebox


face_cascade = cv2.CascadeClassifier(r'D:\Jithu Pgm working\Jithu\Vishva_programs\OPENCV_PLAY\XML\haarcascade_frontalface_default.xml')
def on_click():
    x=e0.get()
    y=e1.get()
    cap=cv2.VideoCapture(0)
    newpath = r'D:\Jithu Pgm working\Jithu\Vishva_programs\OPENCV_PLAY\face-recognition-using-deep-learning-master\face-recognition-using-deep-learning-master\dataset\Our Batch'
    newpath+=f"\{y}_{x}"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
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
        root.destroy()
    else:
        messagebox.showwarning("Warning", "The name and ID is already exist!\nGive a unique name and ID")



root=Tk()
e0=Entry(root,width=25)
e0.pack(pady=15)
e0.insert(0,"Enter the name of the person")

e1=Entry(root,width=25)
e1.pack(pady=15)
e1.insert(0,"Enter the ID")


buttonx=Button(root,text="submit",command=on_click)
buttonx.pack(pady=15)
buttonx.focus
root.geometry("500x500")
root.mainloop()