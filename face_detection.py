import cv2
import numpy as np

pad_factor = .25
out_folder = 'video_faces/'
face_path = "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(face_path)

font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture('8.mov')
flip = True

i = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if flip:
        frame = cv2.flip(frame, -1)
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(
        grey_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )


    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        padx = int(pad_factor * w)
        pady = int(pad_factor * h)
        cv2.rectangle(grey_frame, (x-padx, y-pady), (x+w+padx, y+h+pady), (255, 0, 0), 3)
        face_colour = frame[y-pady:y+h+pady, x-padx:x+w+padx]
        cv2.putText(grey_frame,'Face',(x, y), font, 2,(255,0,0),5)
        cv2.imwrite(out_folder + str(i) + ".jpg",face_colour)
        i = i + 1




    cv2.putText(frame,'Number of Faces : ' + str(len(faces)),(40, 40), font, 1,(255,0,0),2)      
    cv2.imshow('Video', grey_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()