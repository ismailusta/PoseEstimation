import cv2
import numpy as np
import poseEstiminationModul as Psm
import time

cap = cv2.VideoCapture("Source/curl3.mp4")
detector = Psm.poseDetector()
count = 0
pTime = 0
dir = 0
while True:
    success, image = cap.read()
    image = cv2.resize(image, (1280, 720))
    image = detector.findPose(image, draw=False)
    lmList = detector.findPosition(image, draw=False)
    if len(lmList) != 0:
        angle = detector.findAngle(image, 11, 13, 15)
        percentage = np.interp(angle, (200, 310), (0, 100))
        bar = np.interp(angle, (220, 310), (650, 100))
        color = (0, 255, 0)
        if percentage == 100:
            color = (0, 0, 255)
            if dir == 0:
                count += 0.5
                dir = 1
        if percentage == 0:
            color = (0, 0, 255)
            if dir == 1:
                count += 0.5
                dir = 0
        cv2.rectangle(image, (1100, 100), (1175, 655), color, 3)
        cv2.rectangle(image, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(image, f'{int(percentage)} %', (1100, 75), 2, 1, color, 4, cv2.FONT_HERSHEY_PLAIN)

        cv2.rectangle(image, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(image, str(int(count)), (65, 355), 5, 4, (255, 0, 0), 10, cv2.FONT_HERSHEY_PLAIN)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(image, f'FPS:{int(fps)}', (10, 50), 1, 2, (255, 0, 0), 2, cv2.FONT_HERSHEY_PLAIN)
    cv2.imshow("Image", image)
    quitKey = cv2.waitKey(1)
    if quitKey == 27:
        break
