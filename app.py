import streamlit as st
import numpy as np
import tensorflow as tf
import requests
from io import BytesIO
from PIL import Image
import h5py

menu = ['Upload_photo', 'Path']

choice = st.sidebar.selectbox('Upload or use URL path', menu)

def load_model():
	model = tf.keras.models.load_model('my_model_checkpoint25.h5')
	return model
model = load_model()

def preprocess_image(img):
	img = tf.image.resize(img,[224,224])
	img = img/255.
	return np.expand_dims(img, axis =0)

def parse_label(label):
	if label == 0:
		return '1.000'
	elif label == 1:
		return '10.000'
	elif label == 2:
		return '100.000'
	elif label == 3:
		return '2.000'
	elif label == 4:
		return '20.000'
	elif label == 5:
		return '200.000'
	elif label ==6:
		return '5.000'
	elif label == 7:
		return '50.000'
	else:
		return '500.000'


def label(img):
	img = preprocess_image(img)
	label= np.argmax(model.predict(img),axis =1)
	label = parse_label(label[0])
	accuracy = np.asscalar(np.max(model.predict(img),axis =1))

	return label, 'Possibility is: {:.2f}%'.format(accuracy*100)


if choice=='Upload_photo':
	uploaded_file = st.file_uploader('Upload your best photo here', ['png', 'jpeg', 'jpg'])
	if uploaded_file is not None:
		image = Image.open(uploaded_file)
		img = tf.keras.preprocessing.image.img_to_array(image)
		st.image(image, caption='Uploaded Image.', use_column_width=True)
		st.write("")
		st.write("Classifying...")
		with st.spinner('classifying....'):
			label= label(img)
			st.write(label)

if choice=='Path':
	path = st.text_input('Your path here...','https://upload.wikimedia.org/wikipedia/vi/thumb/7/7c/%C4%90%E1%BB%93ng_b%E1%BA%A1c_5000_%C4%91%E1%BB%93ng.jpg/300px-%C4%90%E1%BB%93ng_b%E1%BA%A1c_5000_%C4%91%E1%BB%93ng.jpg')
	if path is not None:
		content = requests.get(path).content
		image = Image.open(BytesIO(content))
		img = tf.image.decode_image(content,expand_animations=False,channels=3)
		st.image(image, caption = 'Classifying Money Image')
		st.write('Predicted class is: ')
		with st.spinner('classifying....'):
			label= label(img)
			st.write(label)
	else: 
		st.write("")
		