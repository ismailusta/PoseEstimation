import cv2
import mediapipe as mp
import time
import math

class poseDetector():

    def __init__(self,staticMode=False,modelComplex=1,smoothLand=True,enableSeg=False,smoothSeg=True,minDetec=0.5,min_track=0.5):
        self.mode = staticMode
        self.complex = modelComplex
        self.smLand = smoothLand
        self.enSeg = enableSeg
        self.smSeg = smoothSeg
        self.detect = minDetec
        self.track = min_track
        self.mDraw = mp.solutions.drawing_utils
        self.mPose = mp.solutions.pose
        self.pose = self.mPose.Pose(self.mode,self.complex,self.smLand,self.enSeg,self.smSeg,self.detect,self.track)

    def findPose(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mDraw.draw_landmarks(img, self.results.pose_landmarks, self.mPose.POSE_CONNECTIONS)
        return img

    def findPosition(self,img,draw=True):
        self.lmList=[]

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), 4, cv2.FONT_HERSHEY_PLAIN)
        return self.lmList
    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
        if angle < 0:
            angle = angle+360

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 4)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 4)
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (255, 0, 0), 2)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 0), 2)
            cv2.circle(img, (x3, y3), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (255, 0, 0), 2)
            cv2.putText(img, str(int(angle)), (x2-50, y2+50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle
def main():
    cap = cv2.VideoCapture(0)
    cTime = 0
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        list = detector.findPosition(img,draw=False)
        if len(list) != 0:
            print(list[14])
            cv2.circle(img, (list[14][1],list[14][2]), 5, (255, 0, 0), 4, cv2.FONT_HERSHEY_PLAIN)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 50), 1, 2, (255, 0, 0), 2, cv2.FONT_HERSHEY_PLAIN)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
