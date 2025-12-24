import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpdraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgrgb)
        if self.result.multi_hand_landmarks:
            for handlms in self.result.multi_hand_landmarks:

                if draw:
                    self.mpdraw.draw_landmarks(img, handlms, self.mphands.HAND_CONNECTIONS)
        return img

    def findPositions(self, img, handno=0, draw=True):
        lmlist = []
        if self.result.multi_hand_landmarks:
            myhand = self.result.multi_hand_landmarks[handno]
            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = (lm.x * w), (lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (int(cx), int(cy)), 7, (0, 255, 0), cv2.FILLED)

        return lmlist

def main():
    ctime = 0
    ptime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPositions(img, draw=False)
        if len(lmlist) != 0:
            print(lmlist[4])
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

        cv2.imshow("img", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()