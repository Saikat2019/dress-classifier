import tensorflow as tf 
from tensorflow import keras
#from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels

print(tf.__version__)

with open('/home/saikat/Documents/diggin_into_DL_CV/dressClassifier/train-images-idx3-ubyte.gz', 'rb') as f:
  train_images = extract_images(f)
with open('/home/saikat/Documents/diggin_into_DL_CV/dressClassifier/train-labels-idx1-ubyte.gz', 'rb') as f:
  train_labels = extract_labels(f)

with open('/home/saikat/Documents/diggin_into_DL_CV/dressClassifier/t10k-images-idx3-ubyte.gz', 'rb') as f:
  test_images = extract_images(f)
with open('/home/saikat/Documents/diggin_into_DL_CV/dressClassifier/t10k-labels-idx1-ubyte.gz', 'rb') as f:
  test_labels = extract_labels(f)


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = 	train_images/255.0
test_images = test_images/255.0

train_images=[train_images[i].squeeze() for i in range(len(train_images))]
test_images=[test_images[i].squeeze() for i in range(len(test_images))]

# plt.figure(figsize=(10,10))
# for i in range(25):
# 	plt.subplot(5,5,i+1)
# 	plt.xticks([])
# 	plt.yticks([])
# 	plt.grid(False)
# 	plt.imshow(train_images[i],cmap=plt.cm.binary)
# 	plt.xlabel(class_names[train_labels[i]])

# plt.show()

#setup the layers

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128,activation=tf.nn.relu),
	keras.layers.Dense(10,activation=tf.nn.softmax)
	])

model.compile(optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])

model.fit(np.array(train_images),np.array(train_labels),epochs=100)

predictions = model.predict(np.array(test_images))

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images[1])
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
  plt.imshow(test_images[i],cmap=plt.cm.binary)

plt.show()