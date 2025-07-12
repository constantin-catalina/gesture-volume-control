import cv2
import time
import numpy as np
import HandTrackingModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

###########################################
wCam, hCam = 640, 480  # Width and height of the camera feed
###########################################

windowName = "Volume Hand Controller"

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam) 

pTime = 0   # Previous time
cTime = 0   # Current time

detector = htm.handDetector(detectionCon=0.7, maxHands=1)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
     IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0
area = 0
colorVol = (255, 0, 0)

while True:
    success, img = cap.read()
    if not success:
            break  # Exit if the frame is not captured successfully
    
    # Find Hand
    img = cv2.flip(img, 1)
    img = detector.findHands(img)  # Detect hands in the image
    lmList, bbox = detector.findPosition(img, draw=False, drawBox=True)  # Get the list of hand landmarks
    
    if len(lmList) != 0:
        # Filter based on size
        wB, hB = bbox[2]-bbox[0], bbox[3]-bbox[1]
        area = (wB * hB) // 100
        
        print(area)
        if 250 < area < 1000:

            # Find the distance between Index and Thumb
            length, img, lineInfo = detector.findDistance(4, 8, img)
            print(length)

            # Convert Volume
            volBar = np.interp(length, [10, 200], [400, 150])
            volPer = np.interp(length, [10, 200], [0, 100])

            # Reduce Resolution to make it smoother
            smoothness = 2
            volPer = smoothness * round(volPer / smoothness)

            # Check which fingers are up
            fingers = detector.fingersUp()
            # print(fingers)

            # Check if pinky is down and set volume
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                colorVol = (0, 255, 0)
            else:
                colorVol = (255, 0, 0)

    # Drawings
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 
                    1, (255, 0, 0), 3)

    cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(img, f'Vol Set: {int(cVol)}', (400, 50), cv2.FONT_HERSHEY_COMPLEX, 1, colorVol, 3)

    # Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 
                1, (255, 0, 0), 3)

    cv2.imshow(windowName, img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break  # Exit if 'q' or 'Esc' is pressed

    if cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1:
        break # Exit if the window is closed

cap.release()
cv2.destroyAllWindows()  # Close all OpenCV windows
