import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

#uploading the file

file = ###put image here

def predict_stage(image_data, model):
    size=(224,224)
    image = ImageOps.fit(image_data,size,Image.ANTIALIAS)
    image_array = np.array(image)
    normalized_image_array = (image_array.astype(np.float32)/127.0) - 1
    data = np.ndarray(shape=(1,224,224,3), dtype=np.float32)
    data[0]=normalized_image_array
    preds=""
    prediction = model.predict(data)
    if n.argmax(prediction)==0:
        print("unripe")
    elif np.argmax(prediction)==1:
        print("over ripe")
    else:
        print("ripe :)")

    return prediction
if file is None:
    print("open camera properly")
else:
    image = Image.open(file)
    model = tf.keras.models.load_model('ripeness.h5')
    prediction = predict_stage(image,model)
    print(prediction)
