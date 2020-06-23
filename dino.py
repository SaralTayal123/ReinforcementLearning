import numpy as np
import cv2 as cv2
from mss import mss
from PIL import Image, ImageEnhance
import keyboard
import pyautogui
import time

import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random
from tqdm import tqdm
from tensorflow.keras.models import model_from_json


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
            pyautogui.press('space')
            # keyboard.press_and_release('space')
        if action == 1:
            pyautogui.press('down')
            # keyboard.press_and_release('down')
        if action == 2:
            junk = 5

        screenshot = self.sct.grab(self.mon)
        img = np.array(screenshot)[:, :, 0]
        processedImg = self._processImg(img)
        state = self._imageBankHandler(processedImg)
        done = self._done(processedImg)
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
        return img

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
        toReturn = np.array(toReturn)
        toReturn = np.reshape(toReturn, (76,384,self.imageBankLength))

        #handle image saving and trimming
        self.imageBank.pop(0)
        self.imageBank.append(img)
        
        return toReturn

    def _getReward(self,done):
        if done:
            return -50
        else: 
            return 1
            return time.time() - self.startTime
        
    def _done(self,img):
        img = np.array(img)
        img  = img[30:50, 180:203, :]
        # print(np.sum(img))
           
        # cv2.imshow("image", img)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()

        # listToCheck = [(18, 137), (19, 152), (16, 165), (16, 178),
        #                (17, 206), (16, 216), (17, 232), (17, 247)]              
        # val = 0
        # for elem in listToCheck:
        #     val += img[elem][0]
        # val = val/8  # avg
        # expectedVal = 0.025179856115107917

        val = np.sum(img)
        expectedVal = 331.9352517985612
        # print("val: ", val)
        # print("Difference: ", np.absolute(val-expectedVal))
        if np.absolute(val-expectedVal) > 1: #seems to work well
            return False
        return True


class Agent:
    def __init__(self):

        self.model = Sequential([
            Conv2D(32, (8,8), input_shape=(76, 384, 4),
                   strides=(2,2), activation='relu'),
            MaxPooling2D(pool_size=(5,5), strides=(2, 2)),
            Conv2D(64, (4,4), activation='relu', strides=(1,1)),
            MaxPooling2D(pool_size=(7, 7), strides=(3, 3)),
            Conv2D(128, (1, 1), strides=(1,1), activation='relu'),
            MaxPooling2D(pool_size=(3,3), strides=(3,3)),
            Flatten(),
            Dense(384, activation='relu'),
            Dense(64, activation="relu", name="layer1"),
            Dense(8, activation="relu", name="layer2"),
            Dense(2, activation="linear", name="layer3"), #2 outputs
        ])
        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))
        # self.model.load_weights("modelwack3.h5")
        self.memory = []
        print(self.model.summary())
        self.xTrain = []
        self.yTrain = []


    def predict(self, state):
        stateConv = state
        # stateConv = np.squeeze(state).reshape(1,-1)
        qval = self.model.predict(np.reshape(stateConv, (1, 76, 384, 4)))
        return qval

    def act(self, state):
        qval = self.predict(state)
        prob = tf.nn.softmax(tf.math.divide((qval.flatten()), 0.7)) #0.7 is the temperature/exploration factor
        # print(np.array(prob))
        action = np.random.choice(range(2), p=np.array(prob))
        return action

    def remember(self, state, nextState, action, reward, done):
        self.memory.append(np.array([state, nextState, action, reward, done]))

    def learn(self):
        self.batchSize = 64

        if len(self.memory) > 100000:
            self.memory = []
            print("trimming memory")
        if len(self.memory) < self.batchSize:
            print("too little info")
            return  # still need to learn, too little memory
        batch = random.sample(self.memory, self.batchSize)
        #check how much time random samples take too

        self.learnBatch(batch)

    def learnBatch(self, batch, alpha=0.9):
        batch = np.array(batch)
        actions = batch[:, 2].reshape(self.batchSize).tolist()
        rewards = batch[:, 3].reshape(self.batchSize).tolist()

        stateToPredict = batch[:, 0].reshape(self.batchSize).tolist()
        nextStateToPredict = batch[:, 1].reshape(self.batchSize).tolist()

        statePrediction = self.model.predict(np.reshape(
            stateToPredict, (self.batchSize, 76, 384, 4)))
        nextStatePrediction = self.model.predict(np.reshape(
            nextStateToPredict, (self.batchSize, 76, 384, 4)))
        statePrediction = np.array(statePrediction)
        nextStatePrediction = np.array(nextStatePrediction)

        for i in range(self.batchSize):
            action = actions[i]
            reward = rewards[i]
            nextState = nextStatePrediction[i]
            qval = statePrediction[i, action]
            statePrediction[i, action] += alpha * (reward + 0.95 * np.max(nextState) - qval)
            # # doubleq^

        self.xTrain.append(np.reshape(
            stateToPredict, (self.batchSize, 76, 384, 4)))
        self.yTrain.append(statePrediction)
        history = self.model.fit(
            self.xTrain, self.yTrain, batch_size=5, epochs=1, verbose=0)
        loss = history.history.get("loss")[0]
        print("LOSS: ", loss)
        self.xTrain = []
        self.yTrain = []


plotX = []

if __name__ == "__main__":
    agent = Agent() #currently agent is configured with only 2 actions
    env = Enviornment()
    env.startGame()    
    for i in range(4):
        state, reward, done = env.reset()
        epReward = 0
        while not done:
            action = agent.act(state)
            # start_time2 = time.time()
            state, reward, done = env.step(0)
            # print("Act Time: ", time.time()-start_time2)

            totalR = 0
            for _ in range(4):  # speeds up learning if you skip frames
                nextState, reward, done = env.step(2) #noop action
                totalR += reward
                if done == True:
                    break
            if done == True:
                print("breaking")
                break
            # start_time2 = time.time()
            agent.remember(state, nextState, action, totalR, done)
            # print("Remember Time: ", time.time()-start_time2)
            state = nextState
            epReward += totalR

        plotX.append(epReward)
        print(epReward)
        agent.learn()
        time.sleep(1)

#####


        

        

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

