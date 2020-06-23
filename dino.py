import numpy as np
import cv2 as cv2
from mss import mss
from PIL import Image, ImageEnhance
import keyboard
import time

import sys
np.set_printoptions(threshold=sys.maxsize)


class Enviornment:
    def __init__(self):
        self.mon = {'top': 380, 'left': 1920, 'width': 1920, 'height': 380}
        self.sct = mss()
        self.counter = 0
        self.startTime = -1
        self.imageBank = []
        self.imageBankLength = 4 #number of frames for the conv net

    def startGame(self):
        #start the game, giving the user a few seconds to click on the chrome tab after starting the code
        for i in reversed(range(3)):
            print("game starting in ", i)
            time.sleep(1)

    def step(self, action):
        if action == 0:
            pass
            keyboard.press_and_release('space')
        if action == 1:
            pass
            keyboard.press_and_release('down')

        screenshot = self.sct.grab(self.mon)
        img = np.array(screenshot)[:, :, 0]
        state = self._processImg(img)
        done = self._done(state)
        reward = self._getReward(done)

        return state, reward, done

    def reset(self):
        self.startTime = time.time()
        return self.step(0)

    def _processImg(self, img):
        img = Image.fromarray(img)
        img = img.resize((384, 76), Image.ANTIALIAS)
        # img = ImageEnhance.Contrast(img).enhance(5)
        img = self._contrast(img)
        img = np.reshape(img, (76,384, 1))
        return self._imageBankHandler(img)

    def _contrast(self,pixvals):
        minval = 32 #np.percentile(pixvals, 2)
        maxval = 171 #np.percentile(pixvals, 98)
        pixvals = np.clip(pixvals, minval, maxval)
        pixvals = ((pixvals - minval) / (maxval - minval))
        # Image.fromarray(pixvals.astype(np.uint8))
        return pixvals

    def _imageBankHandler(self, img):
        while len(self.imageBank) < (self.imageBankLength): 
            self.imageBank.append(img)

        bank = [] + self.imageBank #easy way to deep copy
        toReturn = [] + bank + img

        #handle image saving and trimming
        self.imageBank.pop(0)
        self.imageBank.append(img)
        
        return toReturn

    def _getReward(self,done):
        if done:
            return -100
        else: 
            return time.time() - self.startTime
        
    def _done(self,state):
        img = np.array(state[0])

        # cv2.imshow("image", img)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()

        sum = np.sum(img)
        listToCheck = [(18, 137), (19, 152), (16, 165), (16,178),
                       (17, 206), (16, 216), (17, 232), (17, 247)]
        val = 0
        for elem in listToCheck:
            val += img[elem][0]
        val = val/8 #avg
        expectedVal = 0.048561151079136694
        
        # print("val: ", val)
        # print("Difference: ", np.absolute(val-expectedVal))
        if np.absolute(val-expectedVal) > 0.0008: #seems to work well
            return False
        return True

#TODO agent class

if __name__ == "__main__":
    env = Enviornment()
    env.startGame()
    for episode in range(1):
        state, reward, done = env.reset()
        while not done:
            state, reward, done = env.step(0)
            print(done)
            print(reward)
            #implement main learning loop here



#############
#############
#############
#############
#############
#############
#############

# mon = {'top': 350, 'left': 1920, 'width': 1920, 'height': 400}

# sct = mss()
# counter = 0


# def on_click(event, x, y, p1, p2):
#     print(x,y)

# while 1:
#     # cv2.imread("./test_screenshot_23.06.2020.png")
    
    

#     screenshot = sct.grab(mon)
#     # img = Image.frombytes("RGB", (screenshot.width, screenshot.height), screenshot.rgb)
#     img = np.array(screenshot)[:, :, 0]
#     img = Image.fromarray(img)
#     img = img.resize((384, 76), Image.ANTIALIAS)
#     img = np.array(img, dtype=np.float32)/255  # normalize the image
#     img = np.reshape(img, (76, 384, 1))
#     cv2.imshow("image", img)
#     cv2.namedWindow('image')
#     cv2.setMouseCallback('image', on_click)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break

