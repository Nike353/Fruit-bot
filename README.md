# Fruit-bot
Repository for the fruit-bot event by techsoc 
## Description about each code
### 1)Ripe_train
We are going to use the transfer learning technique to build our model. Transfer learning is a technique of taking a model that is trained on a larger dataset and applying its knowledge to train a much smaller dataset. In our case, we are going to use the pre-trained model Vgg16 and convert it to classify between three categories.


### 2) Ripe_Detect
This is a rospy code which predicts whether the banana in the image is ripe,unripe or over ripe. This will be a node in ros architecture and will publish information to ripe_info topic.

### 3) Fruit_detect 
This is a rospy code which predicts whether the image contains a banana or not based on the model which is trained using the fruit_train code. This will be a node in ROS architecture and will publish the info of banana detection to detect topic.

### 4) Weed_detect 
This is a rospy code which subscribes to the /detect topic and when banana is not detected the weed detection starts and detects weed. In the orignal robot this node will publish the weed detection information in /weed topic but the code given in this repo doesn't include that. 

### 5)Fruit_train
This code helps us to train using kaggel datatset called Fruit-360 using pretrained weights.
