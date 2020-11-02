# TensorFlow and tf.keras
# Runs fine in Python 2
# Make a mininum of 2 changes to parameters.  Indicate what happened to 
# performance and why you think this happened.  Adding more than 2 is
# fine if the analysis is good.  No analysis (just better accuracy
# does not provide a better grade).

# You want to look at keras.io (or at tensorflow https://www.tensorflow.org/) to find the syntax needed. 
import tensorflow as tf
#import keras

#from tf.keras import backend as K
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import metrics
from tensorflow.python.keras import losses
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt



print(tf.__version__)
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0
#print("x_train shape:", train_images.shape, "labels train shape:", train_labels.shape)


test_images = test_images / 255.0
ntrain=train_images[10000:30000]
ntlabels=train_labels[10000:30000]

# Can modify anything BELOW this comment
# See Keras.io for information on Keras
# Fashion mnist info https://github.com/zalandoresearch/fashion-mnist
# If you want to use convolutional layers you have to do something like that below (essentially the shape
# and number of channels.  First argument is number of examples.
#ntrain = ntrain.reshape(ntrain.shape[0], 28, 28, 1)
#test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

#Shivani Singh
#Since I am using convolutional layers in this project so I am reshaping the input to the appropriate size to feed this into our model.
ntrain = ntrain.reshape(ntrain.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

model = tf.keras.Sequential([
	#Shivani Singh
	#Convolutional neural network(CNN) is one of he most popular deep learning algorithm which helps in differentiating one image from other. 
	#CNN helps in transforming the image into something which is easier to process while retaining its original features which helps in classifying the image.
	#I am using CNN into my model because this will help in identifying which image belongs to which class and therefore increasing the accuracy of the whole model over the test data.
	# In the Convolutional layer, I have defined kernel as the matrix of [2,2]
	#This first convolutional layer will help in extracting the low level features from the fashion-mnist dataset. In this first layer I have used 64 feature detectors.	
	tf.keras.layers.Conv2D(kernel_size=[2, 2], filters=64,  padding='same', activation='relu', input_shape=(28, 28, 1)), 
	# here I have applied max pooling so what it can extract the maximum value from the segment of the image which is covered by the kernel at that time. 
	tf.keras.layers.MaxPooling2D(pool_size=2),
	#Shivani Singh
	#I have added dropout by 20% in my model to prevent the over-fitting of this layer into my fashion-mnist training data this helped in increasing the accuracy.
	tf.keras.layers.Dropout(0.2), 
	
	#Shivani Singh
	#This second convolutional layer will help in extracting the high level features from the fashion-mnist dataset 
	#which will help in classifying the images from the fashion-mnist dataset thus improving the accuracy.
	tf.keras.layers.Conv2D(kernel_size=[2, 2], filters=32,  padding='same', activation='relu'),
	# Again I have applied max pooling to extract the maximum value after 2nd convolutional layer.
	tf.keras.layers.MaxPooling2D(pool_size=2),  
	#Shivani Singh
	#Again I have added dropout by 20% in my model to prevent the over-fitting of this layer into my fashion-mnist training data this helped in increasing the accuracy.
	tf.keras.layers.Dropout(0.2),
	
    tf.keras.layers.Flatten(),
	#Increased number of nodes from 50 to 128 to make my model deeply connected. 
	tf.keras.layers.Dense(128, activation= tf.nn.relu),
	#Shivani Singh
	# Added dropout by 30% in my model to prevent the over-fitting of this  dense layer into my fashion-mnist training data. this helped in increasing the accuracy.
	tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=[metrics.sparse_categorical_accuracy]) 


#model.fit(train_images, train_labels, epochs=10)
#Shivani Singh
#Here I have increased the epochs number from 10 to 20 as this will help in making my model more optimal over training data. 
model.fit(ntrain, ntlabels, epochs=20)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
#Shivani Singh
# After appliying convolutional neural network it becomes deep neural network as all the nodes are densely connected and also there are many hidden layers in this models. 
#After applying these modification I am getting accuracy somewhere between .897 (89.7%) - .901 (90.1 %)
