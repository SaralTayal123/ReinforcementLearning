# import cv2 as cv2



# cap = cv2.VideoCapture(0)
# cap.set(3, 320)
# cap.set(4, 240)



# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     # Display the resulting frame
#     pic = np.array(frame)
#     print("Size: ", np.shape(pic))
#     cv2.imshow('preview',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

###
# import serial
# ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
# ser.write(50)

# ser.flush()

# if ser.in_waiting > 0:
#     line = ser.readline()
#     print("INCOMING DATA IS", line)


# ########

import numpy as np
import cv2 as cv2
from mss import mss
from PIL import Image, ImageEnhance, ImageOps
import keyboard
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

#RPI CODE:
import serial
import RPi.GPIO as GPIO  # Import Raspberry Pi GPIO library
GPIO.setwarnings(False)  # Ignore warning for now
GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering
aiMode = 10
arduinoControl = 12
# Set pin 10 to be an input pin and set
GPIO.setup(aiMode, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(arduinoControl, GPIO.out)


class Agent:
    def __init__(self):
        #This is the actual Neural net
        model = Sequential([
            Conv2D(32, (8, 8), input_shape=(76, 384, 4),
                   strides=(2, 2), activation='relu'),
            MaxPooling2D(pool_size=(5, 5), strides=(2, 2)),
            Conv2D(64, (4, 4), activation='relu', strides=(1, 1)),
            MaxPooling2D(pool_size=(7, 7), strides=(3, 3)),
            Conv2D(128, (1, 1), strides=(1, 1), activation='relu'),
            MaxPooling2D(pool_size=(3, 3), strides=(3, 3)),
            Flatten(),
            Dense(384, activation='relu'),
            Dense(64, activation="relu", name="layer1"),
            Dense(8, activation="relu", name="layer2"),
            Dense(3, activation="linear", name="layer3"),
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
        self.ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)

    def act(self, img):
        processedImg = self._processImg(img)
        state = self._imageBankHandler(processedImg)
        action = self.predict(state)
        self.ser.write(str(action).encode('utf-8'))
        return action

    def remember(self, state, action):
        state = self._processImg(state)
        self.location = location
        self.memory.append(np.array([state, action]))

    #TODO: finish funciton
    def generateData(self, batch):
        return None
        #image flip
        #change brightness

    def learn(self):
        batch = np.array(self.memory)
        batch = self.generateData(batch)
        states = batch[:, 0].tolist()
        actions = batch[:, 1].tolist()

        history = self.model.fit(
            states, actions, batch_size=32, epochs=1, verbose=0)
        loss = history.history.get("loss")[0]
        print("LOSS: ", loss)
        self.loss.append(loss)
        self.xTrain = []
        self.yTrain = []

    def _processImg(self, img):
        img = Image.fromarray(img)
        img = self._contrast(img)

        #You can use the following open CV code segment to test your in game screenshots
        # cv2.imshow("image",img)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()

        img = np.reshape(img, (240, 320, 3))
        return img

    def _contrast(self, pixvals):
        minval = 32  # np.percentile(pixvals, 2)
        maxval = 171  # np.percentile(pixvals, 98)
        pixvals = np.clip(pixvals, minval, maxval)
        pixvals = ((pixvals - minval) / (maxval - minval))
        return pixvals

    def _imageBankHandler(self, img):
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
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 320)
        self.cap.set(4, 240)
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
        self.ser.flush()

        if self.ser.in_waiting > 0:
            line = self.ser.readline().decode('utf-8').rstrip()
            print(line)

        return line


agent = Agent()  # currently agent is configured with only 2 actions
while True:  # CHANGE to while button pressed or something
    if GPIO.input(aiMode) == GPIO.HIGH:
        GPIO.output(arduinoControl, GPIO.HIGH)  # TODO: FLIP LOGIC
        state = agent.getState()
        action = agent.observeAction()
        agent.remember(state, action)
    else:
        GPIO.output(arduinoControl, GPIO.HIGH)  # TODO: FLIP LOGIC
        agent.learn()
        agent.model.save_weights("selfdrive.h5")
        while GPIO.input(10) == GPIO.LOW:
            state = agent.getState()
            action = agent.act(state)
