import numpy as np
import cv2
from datetime import datetime
import os, sys
import json

def readJson(name):
    with open(name) as handle:
        objdump = json.loads(handle.read())
    return objdump

def saveToJSON(obj, name):
    JSON = json.dumps(obj)
    f = open(name,"w")
    f.write(JSON)
    f.close()

if len(sys.argv) < 2:
    msg = "Você deve informar o nome do voluntário na linha de comando"
    raise ValueError(msg)

def rotate(image):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    # rotate the image by 180 degrees
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

name = sys.argv[1] #input("Enter name: ")
names = readJson("names_ids.json")
if name in names.values():
    id = list(names.keys())[list(names.values()).index(name)]
    faces = os.listdir('dataset')
    count = len([e for e in faces if e.startswith("User.{}".format(id))])+1
else:
    id = len(names.keys())+1
    names[id] = name
    count = 0

ID = id
#os.mkdir("dataset/User_%s" % ID)

faceCascade = cv2.CascadeClassifier('./haarcascade_frontalcatface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height
startTime = datetime.now()


while(True):
    now_time = str(datetime.now() - startTime).split('.')[0]

    ret, img = cap.read()
    img = cv2.flip(img, -1)
    img = rotate(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(ID) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, now_time, (0,30), font, 1, (0,0,0))
    cv2.imshow('video',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

saveToJSON(names, "names_ids.json")
cap.release()
cv2.destroyAllWindows()
