import numpy as np
import cv2 as cv2
from mss import mss
from PIL import Image, ImageEnhance
import keyboard
# import pyautogui
import time
import tqdm as tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf                                                               
import random
from tqdm import tqdm
from tensorflow.keras.models import model_from_json

class Agent:
    def __init__(self):

        self.model = Sequential([
            Conv2D(32, (8,8), input_shape=(76, 384, 3),
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
            Dense(3, activation="linear", name="layer3"), #2 outputs
        ])
        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))
        # self.model.load_weights("DinoGameredux.h5")
        self.memory = []
        # print(self.model.summary())
        self.xTrain = []
        self.yTrain = []
        self.loss = []


    def predict(self, state):
        stateConv = state
        # stateConv = np.squeeze(state).reshape(1,-1)
        qval = self.model.predict(np.reshape(stateConv, (1, 76, 384, 3)))
        return qval

    def act(self, state):
        qval = self.predict(state)
        # prob = tf.nn.softmax(tf.math.divide((qval.flatten()), 0.6)) #0.7 is the temperature/exploration factor
        # print(np.array(prob))
        z = np.random.random()
        if z > 0.1:
            # print(np.argmax(qval.flatten()))
            return np.argmax(qval.flatten())
        else:
            return np.random.choice(range(3))
        # action = np.random.choice(range(3), p=np.array(prob))
        # return action

    def remember(self, state, nextState, action, reward, done):
        self.memory.append(np.array([state, nextState, action, reward, done]))

    def learn(self):
        self.batchSize = 128

        if len(self.memory) > 100000:
            self.memory = []
            print("trimming memory")
        if len(self.memory) < self.batchSize:
            print("too little info")
            return  # still need to learn, too little memory
        batch = random.sample(self.memory, self.batchSize)
        #check how much time random samples take too

        self.learnBatch(batch)

    def learnBatch(self, batch, alpha=0.8):
        batch = np.array(batch)
        actions = batch[:, 2].reshape(self.batchSize).tolist()
        rewards = batch[:, 3].reshape(self.batchSize).tolist()

        stateToPredict = batch[:, 0].reshape(self.batchSize).tolist()
        nextStateToPredict = batch[:, 1].reshape(self.batchSize).tolist()

        statePrediction = self.model.predict(np.reshape(
            stateToPredict, (self.batchSize, 76, 384, 3)))
        nextStatePrediction = self.model.predict(np.reshape(
            nextStateToPredict, (self.batchSize, 76, 384, 3)))
        statePrediction = np.array(statePrediction)
        nextStatePrediction = np.array(nextStatePrediction)

        for i in range(self.batchSize):
            action = actions[i]
            reward = rewards[i]
            nextState = nextStatePrediction[i]
            qval = statePrediction[i, action]
            if reward < -5: 
                statePrediction[i, action] = reward
            else:
                statePrediction[i, action] += alpha * (reward + 0.95 * np.max(nextState) - qval)
            # # doubleq^

        self.xTrain.append(np.reshape(
            stateToPredict, (self.batchSize, 76, 384, 3)))
        self.yTrain.append(statePrediction)
        history = self.model.fit(
            self.xTrain, self.yTrain, batch_size=5, epochs=1, verbose=0)
        loss = history.history.get("loss")[0]
        print("LOSS: ", loss)
        self.loss.append(loss)
        self.xTrain = []
        self.yTrain = []


class Enviornment:
    def __init__(self):
        self.mon = {'top': 243, 'left': 0, 'width': 1366, 'height': 270}
        # self.mon = {'top': 380, 'left': 0, 'width': 1920, 'height': 380}
        # self.mon = {'top': 1000, 'left': 0, 'width': 3840, 'height': 760}
        self.sct = mss()
        self.counter = 0
        self.startTime = -1
        self.imageBank = []
        self.imageBankLength = 3 #number of frames for the conv net
        self.actionMemory = 2 #init as 2 to show no action taken   
        #image processing
        self.ones = np.ones((76,384,3))
        self.zeros = np.zeros((76,384,3))  
        self.zeros1 = np.zeros((76,384,3))
        self.zeros2 = np.zeros((76,384,3))
        self.zeros3 = np.zeros((76,384,3))
        self.zeros1[:,:,0] = 1
        self.zeros2[:,:,1] = 1
        self.zeros3[:,:,2] = 1

    def startGame(self):
        #start the game, giving the user a few seconds to click on the chrome tab after starting the code
        for i in reversed(range(3)):
            print("game starting in ", i)
            time.sleep(1)

    def step(self, action):        
        actions ={
            0: 'space',
            1: 'down'
        }            
        if action != self.actionMemory:
            if self.actionMemory != 2:
                keyboard.release(actions.get(self.actionMemory))
            if action != 2:
                keyboard.press(actions.get(action))
        self.actionMemory = action

        screenshot = self.sct.grab(self.mon)
        img = np.array(screenshot)[:, :, 0]
        processedImg = self._processImg(img)
        state = self._imageBankHandler(processedImg)
        done = self._done(processedImg)
        reward = self._getReward(done)
        return state, reward, done

    def reset(self):
        self.startTime = time.time()
        keyboard.press("space")
        time.sleep(0.5)
        keyboard.release("space")
        return self.step(0)

    def _processImg(self, img):
        img = Image.fromarray(img)
        img = img.resize((384, 76), Image.ANTIALIAS)
        # img = ImageEnhance.Contrast(img).enhance(5)
        img = self._contrast(img)
        img = np.reshape(img, (76,384))
        return img

    def _contrast(self,pixvals):
        minval = 32 #np.percentile(pixvals, 2)
        maxval = 171 #np.percentile(pixvals, 98)
        pixvals = np.clip(pixvals, minval, maxval)
        pixvals = ((pixvals - minval) / (maxval - minval))
        # Image.fromarray(pixvals.astype(np.uint8))
        return pixvals

    def _imageBankHandler(self, img):
        # timeTest = time.time()
        img = np.array(img)
        # cv2.imshow("image", img)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        while len(self.imageBank) < (self.imageBankLength): 
            self.imageBank.append(np.reshape(img,(76,384,1)) * self.ones)

        
        bank = np.array(self.imageBank)
        toReturn = self.zeros
        img1 = (np.reshape(img,(76,384,1)) * self.ones)  * self.zeros1
        img2 = bank[0] * self.zeros2
        img3 = bank[1] * self.zeros3


        toReturn = np.array(img1 + img2 + img3)
        # toReturn = np.reshape(toReturn, (76,384,4))
        

        self.imageBank.pop(0)
        self.imageBank.append(np.reshape(img,(76 ,384,1)) * self.ones)

        # cv2.imshow("image", np.reshape(toReturn[:,:,0], (76,384,1)))
        # if cv2.waitKey(25) & 0xFF == ord('q'): 
        #     cv2.destroyAllWindows()

        # print("bank Time for loop: ", time.time()-timeTest)

        return toReturn

    def _getReward(self,done):
        if done:
            return -10
        else: 
            return 1
            return time.time() - self.startTime
        
    def _done(self,img):
        img = np.array(img)
        img  = img[30:50, 180:203]
        # cv2.imshow("image",img)
        # if cv2.waitKey(25) & 0xFF == ord('q'): 
        #     cv2.destroyAllWindows()

        val = np.sum(img)
        expectedVal = 331.9352517985612
        # print("val: ", val)
        # print("Difference: ", np.absolute(val-expectedVal))
        if np.absolute(val-expectedVal) > 15: #seems to work well
            return False
        return True

plotX = []
# while True:
if __name__ == "__main__":
    agent = Agent() #currently agent is configured with only 2 actions
    env = Enviornment()
    env.startGame()    
    for i in tqdm(range(1500)):
        state, reward, done = env.reset()
        epReward = 0
        done = False
        episodeTime = time.time()
        stepCounter = 0
        while not done:
            # startTime = time.time()
            action = agent.act(state)
            nextState, reward, done = env.step(action)
            agent.remember(state, nextState, action, reward, done)
            if done == True:
                print("breaking")
                break
            state = nextState
            stepCounter += 1
            # print("episode time: ", time.time()-startTime)
            # print('\n')

        #post episode
        if stepCounter != 0:
            print("Avg Frame-Rate: ", 1/((time.time()-episodeTime)/stepCounter))
        plotX.append(reward)
        print(reward)
        agent.learn()


       
        if i % 20 == 0:
            agent .model.save_weights ("DinoGameSpeed.h5")
            print( "Saved model to disk")                    
# 
            # print("Time action prediction : ", time.time()-start_time2)
            # start_time2 = time.time()