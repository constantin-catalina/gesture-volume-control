import cv2
import mediapipe as mp
import time

class handDetector():
    
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img
    
    def finPosition(self, img, handNo=0, draw=True):
        
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        return lmList


def main():
    pTime = 0   # Previous time
    cTime = 0   # Current time
    cap = cv2.VideoCapture(0)

    detector = handDetector()

    while True:
        success, img = cap.read()
        if not success:
            break  # Exit if the frame is not captured successfully

        img = detector.findHands(img)
        lmList = detector.finPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break  # Exit if 'q' or 'Esc' is pressed

        if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            break # Exit if the window is closed

    cap.release()
    cv2.destroyAllWindows()  # Close all OpenCV windows


if __name__ == "__main__":
    main()