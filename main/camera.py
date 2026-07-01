
import cv2
from main.camera_utils import open_camera

cam = open_camera()

if not cam.isOpened():
    print("Camera haifunguki")
    exit()

while True:
    ret, frame = cam.read()

    if not ret:
        print("Frame haipatikani")
        break

    display_frame = cv2.flip(frame, 1)
    cv2.imshow("Camera Test", display_frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()