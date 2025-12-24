import cv2
import time
import numpy as np
import handtrackingmodule as htm
import math
import mediapipe as mp
import cv2
from pycaw.pycaw import AudioUtilities


device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
device.FriendlyName
# print(f"- Muted: {bool(volume.GetMute())}")
# print(f"- Volume level: {volume.GetMasterVolumeLevel()} dB")
volrange = volume.GetVolumeRange()
minvol = volrange[0]
maxvol = volrange[1]
volper = 0

mpdraw = mp.solutions.drawing_utils
mpface = mp.solutions.face_mesh
mpmesh = mpface.FaceMesh()
dot_color = (0, 255, 0)
mesh_color = (0, 255, 0)
drawing_spec_dots = mpdraw.DrawingSpec(color=dot_color, thickness=1, circle_radius=2)
drawing_spec_mesh = mpdraw.DrawingSpec(color=mesh_color, thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)
wcam , hcam = 640, 480
cap.set(3, wcam)
cap.set(4, hcam)
ptime = 0
detector = htm.handDetector(detectionCon=0.7)
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.rectangle(img, (50, 150), (85, 400), (255, 255, 255), cv2.FILLED)
    faces = mpmesh.process(imgRGB)
    if faces.multi_face_landmarks:
        for facelms in faces.multi_face_landmarks:
            mpdraw.draw_landmarks(img, facelms, mpface.FACEMESH_TESSELATION, drawing_spec_dots, drawing_spec_mesh)
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    img = detector.findHands(img)
    lmlist = detector.findPositions(img)
    if len(lmlist) != 0:
        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cv2.circle(img, (int(cx), int(cy)), 15, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (int(x1), int(y1)),15, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (int(x2), int(y2)),15, (0, 255, 0), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        vol = np.interp(length, [50, 250], [minvol, maxvol])
        volbar = np.interp(length, [50, 250], [400, 150])
        volper = np.interp(length, [50, 250], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)
        cv2.rectangle(img, (50, int(volbar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f"{int(volper)}%", (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        if length<50:
            cv2.circle(img, (int(cx), int(cy)), 15, (0, 0, 255), cv2.FILLED)
    cv2.putText(img, f"FPS: {int(fps)}", (40, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
    cv2.imshow("volume control", img)
    cv2.waitKey(1)