# Fruit-bot
Repository for the fruit-bot event by techsoc 
## Description about each code
### 1)Ripe_train


### 2) Ripe_Detect
This is a rospy code which predicts whether the banana in the image is ripe,unripe or over ripe. This will be a node in ros architecture and will publish information to ripe_info topic.

### 3) Fruit_detect 
This is a rospy code which predicts whether the image contains a banana or not based on the model which is trained using the fruit_train code. This will be a node in ROS architecture and will publish the info of banana detection to detect topic.

### 4) Weed_detect 
This is a rospy code which subscribes to the /detect topic and when banana is not detected the weed detection starts and detects weed. In the orignal robot this node will publish the weed detection information in /weed topic but the code given in this repo doesn't include that. 

