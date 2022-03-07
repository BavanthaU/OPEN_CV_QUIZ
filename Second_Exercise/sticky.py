#!/usr/bin/python3
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def model_func(x, a, k, b):
    return a * np.exp(-k * x) + b

def exponential_multi(x, a, k1, b, k2, c):
    return a * np.exp(x * k1) + b * np.exp(x * k2) + c

# x is the raw distance y is the value in cm
X = np.array([967, 723, 503, 361, 297, 248, 215, 191, 168, 156, 137, 128, 117, 111, 101, 95, 91, 85, 80, 75, 71, 68, 66, 63, 61])
Y = np.array([7, 10, 15, 20, 25, 30, 35, 40, 45, 50 ,55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125])

plt.plot(X, Y, label="line 1")

p0 = (1., 1.e-5, 1., 1.e-5, 1.)  # starting search koefs
# opt, pcov = curve_fit(model_func, X, Y, p0)
opt, pcov = curve_fit(exponential_multi, X, Y, p0)
# a, k, b = opt
aCF, k1CF, bCF, k2CF, cCF = opt

print(aCF, k1CF, bCF, k2CF, cCF)

Y2 = exponential_multi(X, aCF, k1CF, bCF, k2CF, cCF)
plt.plot(X, Y2, label="line 2")

# plt.legend()
# plt.show()

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080) #don't change
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) #don't change
cap.set(cv2.CAP_PROP_EXPOSURE, 40) 

colors = { 
    "yellow"    : {"low" : (24, 40, 150), "upper" :(40, 255, 255)},
    "pink"      : {"low" : (162, 40, 0), "upper" :(175, 255, 255)},
    "orange"    : {"low" : (10, 40, 0), "upper" :(21, 255, 255)},
    "green"     : {"low" : (40, 40, 0), "upper" :(55, 255, 255)}
}

while True:
    #readin the Stream
    success, img = cap.read()

    #convert it to HSV color space for easy threshoulding
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    result = img.copy()

    for color in colors.keys():
        thresh = cv2.inRange(hsv, colors[color]["low"], colors[color]["upper"])

        #remove noise
        #bluring filter
        clean = cv2.medianBlur(thresh,5)

        #apply morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

        # get external contours 
        contours = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03* peri, True)
            
            if len(approx) == 4:
                rot_rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rot_rect)
                box = np.int0(box)

                # center
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(result, (cX, cY), 4, (234,255,0))

                distance = int(math.sqrt((box[1][0] - box[0][0]) ** 2 + (box[1][1] - box[0][1]) ** 2))
                # print("Distance --> ", distance)
                distanceCM = aCF * math.exp(distance * k1CF) + bCF * math.exp(distance * k2CF) + cCF
                print(distanceCM)

                # draw rotated rectangle on copy of img
                cv2.drawContours(result, [box], 0, (255, 255, 255), 3)
                # cv2.line(result, (cX, cY), box[0]+(10,10), (234,255,0), 1)
                cv2.putText(result, (str(round(distanceCM,1)) + " CM"), box[0], cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (234,255,0), 1, cv2.LINE_AA)
            else:
                print("not a rectangle")

    cv2.imshow("Out", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

