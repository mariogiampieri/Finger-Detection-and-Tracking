import cv2
import numpy as np


def main():
    # Blue Color Object Range
    blue_low = np.array([110, 100, 100])
    blue_high = np.array([130, 255, 255])

    # Green Color Object Range
    green_low = np.array([50, 100, 100])
    green_high = np.array([70, 255, 255])

    # Red Color Object Range
    red_low = np.array([0, 100, 100])
    red_high = np.array([10, 255, 255])

    windowOrignal = "Orignal Live Feed"
    windowMasked = "Masked Window Feed"
    windowColorObject = "Color Object Tracked"

    capture = cv2.VideoCapture(0)

    if capture.isOpened():
        flag, frame_bgr = capture.read()
    else:
        flag = False

    while flag:

        flag, frame_bgr = capture.read()

        blurred = cv2.GaussianBlur(frame_bgr, (11, 11), 0)
        frame_hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(frame_hsv, blue_low, blue_high)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cv2.imshow(windowMasked, mask)
        cv2.imshow(windowOrignal, frame_bgr)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    main()