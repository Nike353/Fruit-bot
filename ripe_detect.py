#!/usr/bin/env python
# license removed for brevity
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import rospy
from std_msgs.msg import String
import cv2

def fruit_detect():
    pub = rospy.Publisher('ripe_info', String, queue_size=10)
    rospy.init_node('ripe_detect', anonymous=True)
    rate = rospy.Rate(1) 
    while not rospy.is_shutdown():
        def predict_stage(image_data, model):
            size=(224,224)
            image = ImageOps.fit(image_data,size,Image.ANTIALIAS)
            image_array = np.array(image)
            normalized_image_array = (image_array.astype(np.float32)/127.0) - 1
            data = np.ndarray(shape=(1,224,224,3), dtype=np.float32)
            data[0]=normalized_image_array
            preds=""
            prediction = model.predict(data)
            if np.argmax(prediction)==0:
                print("unripe")
            elif np.argmax(prediction)==1:
                print("over ripe")
            else:
                print("ripe :)")

            return prediction

        cap =cv2.VideoCapture(0)
        success , img = cap.read() 
        model = tf.keras.models.load_model('ripeness.h5')
        prediction =  predict_stage(img,model)

        pub.publish(prediction)
        rate.sleep()

if __name__ == '__main__':
    try:
        fruit_detect()
    except rospy.ROSInterruptException:
        pass