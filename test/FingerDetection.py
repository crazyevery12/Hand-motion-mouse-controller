from datetime import datetime
import cv2
import numpy as np
import pyautogui
import math
from keras.models import load_model

classes = 'NONE ONE TWO THREE FOUR FIVE'.split()
model = load_model('model_6cat.h5')



class FingerDetection:
    """
    Class that tracks the number of occurrences ("counts") of an
    arbitrary event and returns the frequency in occurrences
    (counts) per second. The caller must increment the count.
    """

    def __init__(self):
        self.frame = None
        self.prediction = 0

    def start(self):
        # self._start_time = datetime.now()
        return self


    def detect(self):

        self.prediction = 1

        self.prediction= classes[np.argmax(model.predict(self.frame)[0])]


        return self.prediction