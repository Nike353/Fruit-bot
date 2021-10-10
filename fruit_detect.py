#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2

def fruit_detect():
    pub = rospy.Publisher('detect', String, queue_size=10)
    rospy.init_node('fruit_detect', anonymous=True)
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
            if 'banana' in np.argmax(prediction):
                prediction = "banana"
            else :
                prediction = "~banana"
            
            return prediction
        
        cap =cv2.VideoCapture(0)
        success , img = cap.read() 
        model = tf.keras.models.load_model('banana.h5')
        prediction =  predict_stage(img,model)

        pub.publish(prediction)
        rate.sleep()

if __name__ == '__main__':
    try:
        fruit_detect()
    except rospy.ROSInterruptException:
        pass