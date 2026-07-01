import os
import cv2


def open_camera(prefer_front=True, camera_indices=None, width=640, height=480):
    """Open the first available camera, preferring the front-facing one when possible."""
    if camera_indices is None:
        camera_indices = [0, 1, 2, 3, 4] if prefer_front else [0, 1, 2, 3, 4]

    backends = []
    if os.name == "nt":
        backends.append(cv2.CAP_DSHOW)
    backends.append(cv2.CAP_ANY)

    for backend in backends:
        for index in camera_indices:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                if prefer_front:
                    facing_prop = getattr(cv2, "CAP_PROP_FACING", None)
                    if facing_prop is not None:
                        cap.set(facing_prop, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                return cap

    fallback_backend = backends[0] if backends else cv2.CAP_ANY
    cap = cv2.VideoCapture(0, fallback_backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap
