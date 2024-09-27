import cv2
import mediapipe as mp
import time

mDraw = mp.solutions.drawing_utils
mPose = mp.solutions.pose
pose = mPose.Pose()

cap = cv2.VideoCapture(0)
cTime=0
pTime=0

while True:
    success,img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)

    if results.pose_landmarks:
        mDraw.draw_landmarks(img,results.pose_landmarks,mPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            cx,cy= int(lm.x * w), int(lm.y * h)
            cv2.circle(img,(cx,cy),5,(255,0,0),4,cv2.FONT_HERSHEY_PLAIN)

    cTime= time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(10,50),1,2,(255,0,0),2,cv2.FONT_HERSHEY_PLAIN)

    cv2.imshow("Image",img)
    cv2.waitKey(1)



