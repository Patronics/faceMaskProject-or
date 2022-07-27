import cv2
import os
#cascPath = os.path.dirname(
#    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPath = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPathEyes = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"

faceCascade = cv2.CascadeClassifier(cascPath)
eyesCascade = cv2.CascadeClassifier(cascPathEyes)


##### TODO: modify to use facemarks based on this tutorial: https://pysource.com/2019/03/25/pigs-nose-instagram-face-filter-opencv-with-python/
##### TODO: align video output projection with input view inspired by this tutorial: https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/

qrCodeDetector = cv2.QRCodeDetector()



video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_filtered = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(gray, 60, 120)
    output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    eyes = eyesCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        cv2.rectangle(output, (x, y), (x + w, y + h),(0,255,0), 2)
    for (x,y,w,h) in eyes:
        cv2.circle(output, (int(x+w/2), int(y+h/2)), int(h/2),(255,0,0), 2)
        # Display the resulting frame
    cv2.imshow('Video', output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()