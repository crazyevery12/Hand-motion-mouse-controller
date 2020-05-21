import cv2
import numpy as np
import pyautogui
from FingerDetection import *
from VideoGet import VideoGet
from VideoShow import VideoShow

black_range = np.array([[0, 0, 0], [192, 192, 192]])
black_area = [100, 1700]

font = cv2.FONT_HERSHEY_SIMPLEX

perform = False
showCentroid = False
isDetecting = False
cursor = [960, 540]
kernel = np.ones((7, 7), np.uint8)


def nothing(x):
    pass


def swap(array, i, j):
    temp = array[i]
    array[i] = array[j]
    array[j] = temp


def makeMask(hsv_frame, color_Range):
    mask = cv2.inRange(hsv_frame, color_Range[0], color_Range[1])
    # Morphosis next ...
    eroded = cv2.erode(mask, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    return dilated


def drawCentroid(vid, color_area, mask, showCentroid):
    contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    l = len(contour)
    area = np.zeros(l)

    for i in range(l):
        if cv2.contourArea(contour[i]) > color_area[0] and cv2.contourArea(contour[i]) < color_area[1]:
            area[i] = cv2.contourArea(contour[i])
        else:
            area[i] = 0

    a = sorted(area, reverse=True)

    for i in range(l):
        for j in range(1):
            if area[i] == a[j]:
                swap(contour, i, j)

    if l > 0:
        # finding centroid using method of 'moments'
        M = cv2.moments(contour[0])
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            center = (cx, cy)
            if showCentroid:
                cv2.circle(vid, center, 5, (0, 0, 255), -1)

            return center
    else:
        # return error handling values
        return (-1, -1)


def chooseAction(pred):
    out = np.array(['none', 'false'])

    if pred == 'ONE':
        out[0] = 'move'
        return out

    else:
        out[0] = -1
        return out


def performAction(black_centroid, action, perform):
    if perform:
        cursor[0] = 4 * (black_centroid[0] - 110)

        cursor[1] = 4 * (black_centroid[1] - 120)

        if action == 'move':
            if black_centroid[0] > 110 and black_centroid[0] < 590 and black_centroid[1] > 120 and black_centroid[
                1] < 390:
                pyautogui.moveTo(cursor[0], cursor[1])
            elif black_centroid[0] < 110 and black_centroid[1] > 120 and black_centroid[1] < 390:
                pyautogui.moveTo(8, cursor[1])
            elif black_centroid[0] > 590 and black_centroid[1] > 120 and black_centroid[1] < 390:
                pyautogui.moveTo(1912, cursor[1])
            elif black_centroid[0] > 110 and black_centroid[0] < 590 and black_centroid[1] < 120:
                pyautogui.moveTo(cursor[0], 8)
            elif black_centroid[0] > 110 and black_centroid[0] < 590 and black_centroid[1] > 390:
                pyautogui.moveTo(cursor[0], 1072)
            elif black_centroid[0] < 110 and black_centroid[1] < 120:
                pyautogui.moveTo(8, 8)
            elif black_centroid[0] < 110 and black_centroid[1] > 390:
                pyautogui.moveTo(8, 1072)
            elif black_centroid[0] > 590 and black_centroid[1] > 390:
                pyautogui.moveTo(1912, 1072)
            else:
                pyautogui.moveTo(1912, 8)


def calibrateColor(color, def_range):
    global kernel
    name = 'Calibrate ' + color
    cv2.namedWindow(name)
    cv2.createTrackbar('Hue', name, def_range[0][0] + 20, 180, nothing)
    cv2.createTrackbar('Sat', name, def_range[0][1], 255, nothing)
    cv2.createTrackbar('Val', name, def_range[0][2], 255, nothing)
    while (1):
        ret, frameinv = cap.read()
        frame = cv2.flip(frameinv, 1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        hue = cv2.getTrackbarPos('Hue', name)
        sat = cv2.getTrackbarPos('Sat', name)
        val = cv2.getTrackbarPos('Val', name)

        lower = np.array([hue - 20, sat, val])
        upper = np.array([hue + 20, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)
        eroded = cv2.erode(mask, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)

        cv2.imshow(name, dilated)

        k = cv2.waitKey(5) & 0xFF
        if k == ord(' '):
            cv2.destroyWindow(name)
            return np.array([[hue - 20, sat, val], [hue + 20, 255, 255]])
        elif k == ord('d'):
            cv2.destroyWindow(name)
            return def_range


def handDetect( prediction):
    """
    Add iterations per second text to lower-left corner of a frame.
    """
    # cv2.putText(frame, '%s' % (prediction), (x+10,y+50), font, 1.0, (245, 210, 65), 2, 1)
    return prediction



def binaryMask(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7,7), 3)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, new = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return new

def configFrame(frame):
    frame = cv2.resize(frame, (300,300))
    roi = binaryMask(frame)
    img = np.float32(roi) / 255.
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=3)
    return img

def main():
    video_getter = VideoGet(0).start()
    video_shower = VideoShow(video_getter.frame).start()
    finger_detect = FingerDetection()
    finger_detect.start()

    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            break

        frame = video_getter.frame
        frame = cv2.flip(frame,1)

        img = configFrame(frame)
        finger_detect.frame = img

        prediction = handDetect(finger_detect.detect())
        print("Prediction: ", prediction)

        video_shower.frame = frame







cap = cv2.VideoCapture(0)
#cv2.namedWindow("Frame")
black_range = calibrateColor("Black", black_range)
#if __name__ == "__main__":
    #main()

""""while(1):
    k = cv2.waitKey(10) & 0xFF
    _, frameinv = cap.read()
    frame = cv2.flip( frameinv, 1)
    hsv = cv2.cvtColor( frame, cv2.COLOR_BGR2HSV)
    b_mask = makeMask( hsv, black_range)

    cv2.imshow('Frame', frame)

    if k == 27:
        break
cv2.destroyWindow()"""

