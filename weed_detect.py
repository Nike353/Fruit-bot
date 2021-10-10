#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import String

def callback(data):
    if data.data == "~banana" :
        cap =cv2.VideoCapture(0)
        success , img = cap.read()
        #cv2.imshow("img",img)
        imghsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        lower=np.array([25,0,30])
        upper=np.array([80,255,255])

        mask =cv2.inRange(imghsv,lower,upper)
            
        imgres = cv2.bitwise_and(img,img,mask=mask)
        imggray = cv2.cvtColor(imgres,cv2.COLOR_BGR2GRAY)
        #cv2.imshow("res",imgres)
        #cv2.imshow("grey",imggray)

        Gaussian = cv2.GaussianBlur(imggray,(7,7),0)
        #cv2.imshow('Gaussian Blurring', Gaussian)


        biblur = cv2.bilateralFilter(Gaussian, 3, 50,50)
        #cv2.imshow('biblur', biblur)
        edges = cv2.Canny(biblur, 100,100)
        kernel = np.ones((5,5), np.uint8)
        img_dilation = cv2.dilate(edges, kernel, iterations=1)
        kernel3 = np.ones((3,3), np.uint8)
        img_erosion = cv2.erode(img_dilation, kernel3, iterations=1)

        cv2.imshow("edges", edges)
        cv2.imshow("erode", img_erosion)
        total= img_erosion.size
        n_white_pix = np.sum(img_erosion == 255)
        percentage =  (n_white_pix/total)*100
        if (percentage > 20):
            print(1)
        else :
            print(0)
        #cv2.waitKey(0)
    
def weed_detect():

    
    rospy.init_node('weed_detect', anonymous=True)

    rospy.Subscriber("detect", String, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    weed_detect()