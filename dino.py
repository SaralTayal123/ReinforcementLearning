import numpy as np
import cv2
from mss import mss
from PIL import Image
import keyboard
import time


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
        for i in reversed(range(5)):
            print("game starting in ", i)
            time.sleep(1)

    def step(self, action):
        if action == 0:
            keyboard.press_and_release('space')
        if action == 1:
            keyboard.press_and_release('down')

        screenshot = self.sct.grab(self.mon)
        img = np.array(screenshot)[:, :, 0]
        state = self._processImg(img)
        done = self._done(state, img)
        reward = self._getReward(done)

        return state, reward, done

    def reset(self):
        self.startTime = time.time()
        return self.step(0)

    def _processImg(self, img):
        img = Image.fromarray(img)
        img = img.resize((152, 768), Image.ANTIALIAS)
        img = np.array(img, dtype=np.float32)/255 #normalize the image
        return self._imageBankHandler(img)


    def _imageBankHandler(self, img):
        while len(self.imageBank) < (self.imageBankLength-1): 
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
            return time.time() - self.startGame
        
    def _done(self,img):
        return 
        #TODO
        #probably look at certain pixels or something

#TODO agent class

if __name__ == "__main__":
    env = Enviornment()
    env.startGame()
    for episode in range(100):
        state, reward, done = env.reset()
        while done == false:
            pass
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

# while 1:
#     screenshot = sct.grab(mon)
#     # img = Image.frombytes("RGB", (screenshot.width, screenshot.height), screenshot.rgb)
#     img = np.array(screenshot)[:, :, 0]
#     cv2.imshow('test', img)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         time.sleep(3)
#         print("pressing space")
#         keyboard.press_and_release('space')
#         counter += 1
#         if counter > 2:
#             cv2.destroyAllWindows()
#             break
