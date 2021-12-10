# MANTHAN
# Here we are using this recognition to identify and differentiate stranger and familiar people and Tkinter for GUI.
# And also provided movement, age and gender detection

- Create dataset of face images
- Detect faces using deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel.
- Since this model is used in many areas, as they are having high accuracy, it is easy to identify the faces in short span of time.
- Extract face embeddings for each face present in the image using pretrained and train those SVM model on the face embeddings to recognize faces using webcam.
- Here the spoofing technique is used, inorder to minimize the false alarm.

# Overview of OpenFace for a single input image
1. Detect faces with a pre-trained models from dlib or OpenCV.
2. Transform the face for the neural network. This repository uses dlib's real-time pose estimation with OpenCV's affine transformation to try to make 
   the eyes and bottom lip appear in the same location on each image.
3. Use a deep neural network to represent (or embed) the face on a 128-dimensional unit hypersphere. 
   The embedding is a generic representation for anybody's face. Unlike other face representations, this embedding has the nice property 
   that a larger distance between two face embeddings means that the faces are likely not of the same person. 
   This property makes clustering, similarity detection, and classification tasks easier than other face recognition techniques 
   where the Euclidean distance between features is not meaningful.
4. Apply clustering or classification techniques to the features to complete the face recognition task.

# How to run the program
 - Create dataset of face images.
 - It will ask whether to read in the file or record live for images
 - Place those face images in dataset folder.
 - Train the SVM model - python train_model.py
 - Before jumping to test the model, if you want to insert any audio file to alert us for the stranger, you have to enter filename to run that audio. More info: https://pythonbasics.org/python-play-sound/ 
 - Test the model - python recognize_video.py
# NOTE
Actually to load or to run the model you need a file called *openface_nn4.small2.v1.t7*
You can download by using this link: https://github.com/pyannote/pyannote-data/blob/master/openface.nn4.small2.v1.t7

For age and gender detection you need .prototxt and .caffemodel file respectively. *Create those files or search them*

## Prerequisites
- Python 3.8.1
- OpenCV
- Tensorflow
```
sudo apt-get install python-opencv
```
More info to install Tensorflow: https://cran2367.medium.com/install-and-setup-tensorflow-2-0-2c4914b9a265


