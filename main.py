from PIL import Image

import streamlit as st

import tensorflow as tf

import numpy as np



def load_model():

	model = tf.keras.models.load_model('./model.keras')

	return model



def predict_class(image, model):

	image = tf.cast(image, tf.float32)

	image = tf.image.resize(image, [32, 32])
 
	image = image/255.0

	image = np.expand_dims(image, axis = 0)

	prediction = model.predict(image)

	return prediction



model = load_model()

st.title('Traffic Sign Classifier')

file = st.file_uploader("Upload an image of a traffic sign.", type=["jpg"])

if file is None:

	st.text('Waiting for upload....')

else:

	slot = st.empty()

	slot.text('Model prediction in progress....')

	test_image = Image.open(file)

	st.image(test_image, caption="Uploaded image", width = 400)

	pred = predict_class(np.asarray(test_image), model)

	class_names = ['Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)', 'No passing', 'No passing for vehicles over 3.5 metric tons', 'Right-of-way at the next intersection',
	'Priority road',
	'Yield',
	'Stop',
	'No vehicles',
	'Vehicles over 3.5 metric tons prohibited',
	'No entry',
	'General caution',
	'Dangerous curve to the left',
	'Dangerous curve to the right',
	'Double curve',
	'Bumpy road',
	'Slippery road',
	'Road narrows on the right',
	'Road work',
	'Traffic signals',
	'Pedestrians',
	'Children crossing',
	'Bicycles crossing',
	'Beware of ice/snow',
	'Wild animals crossing',
	'End of all speed and passing limits',
	'Turn right ahead',
	'Turn left ahead',
	'Ahead only',
	'Go straight or right',
	'Go straight or left',
	'Keep right',
	'Keep left',
	'Roundabout mandatory',
	'End of no passing',
	'End of no passing by vehicles over 3.5 metric']

	print(np.argmax(pred))

	
	print(pred)
	result = class_names[np.argmax(pred)]


	output = 'The image is a ' + result

	slot.text('Done')

	st.success(output)