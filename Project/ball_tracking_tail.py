from collections import deque
import numpy as np
import imutils
import cv2
import time


greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque()

camera = cv2.VideoCapture("video.mp4")

# allow the camera or video file to warm up
time.sleep(2.0)



while True:
    (grabbed, frame) = camera.read()


    if not grabbed: 
        break

    frame = imutils.resize(frame, width=1000)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

    pts.appendleft(center)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
         #if either of the tracked points are None, ignorethem
        if pts[i - 1] is None or pts[i] is None:
            continue
        
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 3)
    
    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)
    

    key = cv2.waitKey(1) & 0xFF

f = open("data_point.txt", "w") 
#I saved the data point to a file called data_points.txt which should be created
#in the same directory that the code is in. You should be able to open it in a notebook
# using something like f = open("data_point.txt", "r") to read from the file. If the 
#notebook is not in the same path as the data file then just replace the file name with the path. 
f.write(str(list(pts)))
f.close()

camera.release()
cv2.destroyAllWindows()

