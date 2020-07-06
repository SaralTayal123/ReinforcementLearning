import RPi.GPIO as GPIO  # Import Raspberry Pi GPIO library
import blynklib
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import time
from PIL import Image, ImageEnhance, ImageOps
import cv2 as cv2
import numpy as np
import pickle
mport pandas as pd


GPIO.setwarnings(False)  # Ignore warning for now
GPIO.setmode(GPIO.BCM)  # Use physical pin numbering
motor = 13
steering = 12
GPIO.setup(motor, GPIO.OUT)
GPIO.setup(steering, GPIO.OUT)
motorServo = GPIO.PWM(motor, 50)
motorServo.start(2.5)
steeringServo = GPIO.PWM(steering, 50)
steeringServo.start(5)


class Agent:
    def __init__(self):
        #This is the actual Neural net
        self.userSteering = 90
        self.aiMode = False
        model = Sequential([
            Conv2D(32, (7, 7), input_shape=(240, 320, 3),
                   strides=(2, 2), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(5, 5), strides=(2, 2), padding='valid'),
            Conv2D(64, (4, 4), activation='relu',
                   strides=(1, 1), padding='same'),
            MaxPooling2D(pool_size=(4, 4), strides=(2, 2), padding='valid'),
            Conv2D(128, (4, 4), strides=(1, 1),
                   activation='relu', padding='same'),
            MaxPooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid'),
            Flatten(),
            Dense(384, activation='relu'),
            Dense(64, activation="relu", name="layer1"),
            Dense(8, activation="relu", name="layer2"),
            Dense(1, activation="linear", name="layer3"),
        ])
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))
        # model.load_weights("DinoGameSpeed4.h5")
        self.model = model
        self.memory = []
        # print(self.model.summary())
        self.xTrain = []
        self.yTrain = []
        self.loss = []
        self.imageBank = []
        self.imageBankLength = 4  # number of frames for the conv net
        #image processing
        self.ones = np.ones((240, 320, 12))
        self.zeros = np.zeros((240, 320, 12))
        self.zeros1 = np.zeros((240, 320, 12))
        self.zeros2 = np.zeros((240, 320, 12))
        self.zeros3 = np.zeros((240, 320, 12))
        self.zeros4 = np.zeros((240, 320, 12))
        self.zeros1[:, :, 0:3] = 1
        self.zeros2[:, :, 3:6] = 1
        self.zeros3[:, :, 6:9] = 1
        self.zeros4[:, :, 9:12] = 1

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 320)
        self.cap.set(4, 240)

    def act(self, img):
        processedImg = self._processImg(img)
        state = self._imageBankHandler(processedImg)
        state = np.reshape(state, (1, 240, 320, 3))
        action = self.model.predict(state)
        steeringServo.ChangeDutyCycle((action * 6)+4)  # range of 4-10
        return action
    #TODO:insert pwm control

    def remember(self, state, action):
        state = self._processImg(state)
        self.memory.append(np.array([state, action]))

    #TODO: finish funciton
    def generateData(self, batch):
        return None
        #image flip
        #change brightness

    def learn(self):
        batch = np.array(self.memory)
        # batch = self.generateData(batch) #TODO: finish integration
        print(np.shape(batch), "Batch shape")
        states = np.array(batch[:, 0].tolist())
        actions = np.array(batch[:, 1].tolist())
        #states = np.reshape(states, (np.shape(batch)[0], 240,320,3))
        print(np.shape(actions), "action shape")
        print(np.shape(states), "STATE SHAPE")
        history = self.model.fit(
            states, actions, batch_size=32, epochs=1, verbose=0)
        loss = history.history.get("loss")[0]
        print("LOSS: ", loss)
        self.loss.append(loss)
        self.xTrain = []
        self.yTrain = []

    def _processImg(self, img):
       # img = Image.fromarray(img)
       # img = self._contrast(img)

        #You can use the following open CV code segment to test your in game screenshots
        # cv2.imshow("image",img)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()

        img = np.reshape(img, (240, 320, 3)) / 255
        return img

    def _contrast(self, pixvals):
        minval = 32  # np.percentile(pixvals, 2)
        maxval = 171  # np.percentile(pixvals, 98)
        pixvals = np.clip(pixvals, minval, maxval)
        pixvals = ((pixvals - minval) / (maxval - minval))
        return pixvals

    def _imageBankHandler(self, img):
        return img
        while len(self.imageBank) < (self.imageBankLength):
            self.imageBank.append(np.reshape(img, (240, 320, 3)) * self.ones)

        bank = np.array(self.imageBank)
        toReturn = self.zeros
        img1 = (img * self.ones) * self.zeros1
        img2 = bank[0] * self.zeros2
        img3 = bank[1] * self.zeros3
        img4 = bank[2] * self.zeros4

        toReturn = np.array(img1 + img2 + img3 + img4)

        self.imageBank.pop(0)
        self.imageBank.append(np.reshape(img, (240, 320, 3)) * self.ones)

        return toReturn
    def getState(self):

        ret, frame = self.cap.read()
        pic = np.array(frame)
        processedImg = self._processImg(pic)
        state = self._imageBankHandler(processedImg)
        return state
        # print("Size: ", np.shape(pic))
        # cv2.imshow('preview',frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    def observeAction(self):
        return (self.userSteering - 4) / 6


agent = Agent()  # currently agent is configured with only 2 actions
BLYNK_AUTH = 'MGbFmANkyThXj6e36bE0JnPDWK7V84Xy'
blynk = blynklib.Blynk(BLYNK_AUTH)
aiMode = False

def servoControl(value):
    steeringServo.ChangeDutyCycle(value)


@blynk.handle_event('write V4')
def write_virtual_pin_handler(pin, value):
    print("value: ",float(value[0]))
    agent.userSteering = float(value[0])
    servoControl(float(value[0]))

@blynk.handle_event('write V2')
def write_virtual_pin_handler(pin, value):
    if value == 1:
        agent.aiMode = False
    else:
        agent.aiMode = True


counter = 0
while True:  # CHANGE to while button pressed or something
    blynk.run()
    if agent.aiMode == False:
        print("Manual Control")
        state = agent.getState()
        action = agent.observeAction()
        if action >= 0: #if its a valid action
            agent.remember(state, action)
            counter += 1
            if counter % 100 == 0:
                filehandler = open('aiMemoryPickle', "wb")
                pickle.dump(agent.memory, filehandler)
                filehandler.close()
    else:
        print("AI Control")
        agent.learn()
        agent.model.save_weights("selfdrive.h5")
        while agent.aiMode == True:
            state = agent.getState()
            action = agent.act(state)
            print("action", action)
                                                                                                                                                                                                                                                                                                            207,1         Bot
