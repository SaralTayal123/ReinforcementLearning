import cv2 as cv2



cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)



while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Display the resulting frame
    pic = np.array(frame)
    print("Size: ", np.shape(pic))
    cv2.imshow('preview',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

###
import serial
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
ser.write(50)

ser.flush()

if ser.in_waiting > 0:
    line = ser.readline()
    print("INCOMING DATA IS", line)
