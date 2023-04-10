# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.
<br>
![230705640-8ad66af5-a17f-48fb-bacb-2fd8121fe51b](https://user-images.githubusercontent.com/93427278/230726342-18ce57e0-6048-4702-b8db-b7159c95fe59.png)
<br>
## Neural Network Model
![230705825-20755f3c-9ada-416d-be61-f416817964af](https://user-images.githubusercontent.com/93427278/230725435-9d05144a-0bde-48c3-90b2-c8f909c32ff9.png)


## DESIGN STEPS

### STEP 1:
import tensorflow and preprocessing libraries
### STEP 2:
Build a CNN model
### STEP 3:
Compile and fit the model and then predict

## PROGRAM
```
Developed by : Vishranthi A
Reg no.: 212221230124
```
```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[0]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
model=keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')
model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))
# Prediction for a single input
img = image.load_img('7.png')
type(img)
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://user-images.githubusercontent.com/93427278/230725998-e3c1b847-d32c-4fbb-81a2-a3effe31a5a7.png)
<br>
![image](https://user-images.githubusercontent.com/93427278/230726017-93a7f949-9a2e-4a86-ada3-d845260b9d82.png)
<br>

### Classification Report

![image](https://user-images.githubusercontent.com/93427278/230726093-2b05deb4-fc69-44e8-9baa-8ea1e72cb201.png)

<br>

### Confusion Matrix

![image](https://user-images.githubusercontent.com/93427278/230726062-ef2a94c4-e87c-4e40-8263-8db9428ebd80.png)
<br>

### New Sample Data Prediction
![image](https://user-images.githubusercontent.com/93427278/230726159-fe6608cb-f827-456e-80c5-6a6f60dc93b8.png)
<br>
![image](https://user-images.githubusercontent.com/93427278/230726182-e81dbd42-ddfa-4a92-83cc-fcac895e36e0.png)


## RESULT
A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed sucessfully.
