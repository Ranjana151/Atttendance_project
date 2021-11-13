import numpy as np
import cv2
import face_recognition
import os
from datetime import datetime

# Get all the images from a directory
path = "ImagesFiles"
images_list = []
name_list = []
myList = os.listdir(path)
# print(myList)

# Reading of the image one by one and  appending images,name  into lists
for cl in myList:
    currImg = cv2.imread(f'{path}/{cl}')
    images_list.append(currImg)
    name_list.append(os.path.splitext(cl)[0])
#print(name_list)

# find encoding of the images known
def findEncodings(images_list):
    encodingList = []
    for img in images_list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodingList.append(encode)
    return encodingList


encodeListknown = findEncodings(images_list)
# print(len(encodeListknown))

# Mark the attendance if name is not present already in file

def markAttendance(name):
    with open("Attendance.csv", "r+") as f:
        present_data = f.readlines()
        name_present = []
        for i in present_data:
            entry = i.split(',')
            name_present.append(entry[0])
        if name not in name_present:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')




cap = cv2.VideoCapture(0)
# Image from webcam
while True:
    success,img = cap.read()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #  webcam image encoding and location of face in webcam image
    facesCurrFrame = face_recognition.face_locations(img)
    encodingCurrFrame = face_recognition.face_encodings(img,facesCurrFrame)

    # Webcam image comparison with known image
    for encodeFace, faceLoc in zip(encodingCurrFrame,facesCurrFrame):
        matches = face_recognition.compare_faces(encodeListknown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListknown,encodeFace)
        #print(faceDis)

        # Matching index of webcam image with known images
        matchIndex = np.argmin(faceDis)

        # Name of matching image
        if matches[matchIndex]:
            name = name_list[matchIndex].upper()
            #print(name)

            # Draw rectangle on face in webcam image

            y1, x2, y2, x1 = faceLoc
            cv2.rectangle(img, (x1, y1), (x2, y2),(0,255,0),2)

            # Put name of matching image on webcam image
            cv2.putText(img, name, (x1, y2+50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            # Mark the attendance
            markAttendance(name)




    cv2.imshow("Image", img)
    cv2.waitKey(1)






