---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region id="t0LoK7eSDi2n" colab_type="text" -->
# Cat-Breed Classifier Using CNN
<!-- #endregion -->

<!-- #region id="ppI-4GeliO9j" colab_type="text" -->
## 1. Aim
 is to buid a classifier to categorize the breed of cats into 4 types:   'Abyssian',   'Munchkin',    'Persian',   'Toyger' on feeding the input image of a cat using Convolutional Neural Network with Tensorflow.

Here is a link to refer a sample CNN implementation using TF: https://www.tensorflow.org/tutorials/images/cnn

<!-- #endregion -->

<!-- #region id="fxMf6TE3j_b0" colab_type="text" -->
## 2. Setup

<!-- #endregion -->

```python id="Eg6qBHBSDi2q" colab_type="code" colab={}
#import the required libraries
import os
import zipfile
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

```

```python id="7sCqqdCQrF6z" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="b83a8705-38ec-44b7-f127-ed82ba9dbaa1" executionInfo={"status": "ok", "timestamp": 1588233934925, "user_tz": -330, "elapsed": 2770, "user": {"displayName": "Manmohan Dogra", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhoL7CtYUEHr9zkw97VpbpdDqYV30gXSuOX9CSBdA=s64", "userId": "04603716801366227129"}}
from google.colab import drive
drive.mount('/content/drive')
```

<!-- #region id="_SVhDjkNkdiQ" colab_type="text" -->
## 3.1 Loading Dataset:
  Let's load the dataset from the directory and store the path in variable.

  Here the hirerarchy is: 
    -
        

1.   Cat Breed Classifier
    1. Cat-Breed-Classifier.ipynb
    2. Dataset
          1. Training:: 
              -Abyssian -Munchkin -Persian -Tygor
          2. Validation::
              -Abyssian -Munchkin -Persian -Tygor


        
        
      
<!-- #endregion -->

```python id="gzKDvUMDDi22" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 158} outputId="6de09355-305c-456a-9638-3b86a7b201e3" executionInfo={"status": "ok", "timestamp": 1588233937298, "user_tz": -330, "elapsed": 859, "user": {"displayName": "Manmohan Dogra", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhoL7CtYUEHr9zkw97VpbpdDqYV30gXSuOX9CSBdA=s64", "userId": "04603716801366227129"}}
#collecting the path for base directory
base_dir='/content/drive/My Drive/Cat-Breed-Classifier/dataset/'
training_dir=os.path.join(base_dir, 'training')
validation_dir=os.path.join(base_dir, 'validation')

train_a_dir=os.path.join(training_dir,'abyssian')
train_m_dir=os.path.join(training_dir,'munchkin')
train_p_dir=os.path.join(training_dir,'persian')
train_t_dir=os.path.join(training_dir,'toyger')

valid_a_dir=os.path.join(validation_dir,'abyssian')
valid_m_dir=os.path.join(validation_dir,'munchkin')
valid_p_dir=os.path.join(validation_dir,'persian')
valid_t_dir=os.path.join(validation_dir,'toyger')


#Let's find out the total number of horse and human images in the directories:
print('total abyssian in training: ', len(os.listdir(train_a_dir)))
print('total munchkin in training: ', len(os.listdir(train_m_dir)))
print('total persian in training: ', len(os.listdir(train_p_dir)))
print('total toyger in training: ', len(os.listdir(train_t_dir)))

print('total abyssian in validation: ', len(os.listdir(valid_a_dir)))
print('total munchkin in validation: ', len(os.listdir(valid_m_dir)))
print('total persian in validation: ', len(os.listdir(valid_p_dir)))
print('total toyger in validation: ', len(os.listdir(valid_t_dir)))

```

<!-- #region id="9JbTT9ilmTi0" colab_type="text" -->
## 3.2 One look at the dataset.

<!-- #endregion -->

```python id="7E-5NpBGnevH" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 178} outputId="0ce8f015-dc4d-42b5-9807-1c87418c76d5" executionInfo={"status": "ok", "timestamp": 1588233938695, "user_tz": -330, "elapsed": 1601, "user": {"displayName": "Manmohan Dogra", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhoL7CtYUEHr9zkw97VpbpdDqYV30gXSuOX9CSBdA=s64", "userId": "04603716801366227129"}}
#let's see the files in the directories
train_a_names = os.listdir(train_a_dir)
print(train_a_names[:10])
train_m_names = os.listdir(train_m_dir)
print(train_m_names[:10])
train_p_names = os.listdir(train_p_dir)
print(train_p_names[:10])
train_t_names = os.listdir(train_t_dir)
print(train_t_names[:10])


validation_a_names = os.listdir(valid_a_dir)
print(validation_a_names[:10])
validation_m_names = os.listdir(valid_m_dir)
print(validation_m_names[:10])
validation_p_names = os.listdir(valid_p_dir)
print(validation_p_names[:10])
validation_t_names = os.listdir(valid_t_dir)
print(validation_t_names[:10])


```

```python id="87jEaF9bmSI6" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 901} outputId="05cf357e-6e84-4ab8-83f0-23eac20248d0" executionInfo={"status": "ok", "timestamp": 1588233943526, "user_tz": -330, "elapsed": 5886, "user": {"displayName": "Manmohan Dogra", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhoL7CtYUEHr9zkw97VpbpdDqYV30gXSuOX9CSBdA=s64", "userId": "04603716801366227129"}}
#Let's see the images present in the dataset.

import matplotlib.image as mpimg
# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 4
next_a_pix = [os.path.join(train_a_dir, fname) 
                for fname in train_a_names[pic_index-4:pic_index]]
next_m_pix = [os.path.join(train_m_dir, fname) 
                for fname in train_m_names[pic_index-4:pic_index]]
next_p_pix = [os.path.join(train_p_dir, fname) 
                for fname in train_p_names[pic_index-4:pic_index]]
next_t_pix = [os.path.join(train_t_dir, fname) 
                for fname in train_t_names[pic_index-4:pic_index]]

for i, img_path in enumerate(next_a_pix+next_m_pix+next_p_pix+next_t_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') 

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

#Abyssian
#Munchkin
#Persian
#Tygor
```

<!-- #region id="khOtZHGTpdfg" colab_type="text" -->
## 4. Building a Model
Let's create our sequential layers. 
Here's a tutorial link for the same :
http://keras.io/layers/convolutional/
<!-- #endregion -->

```python id="V6lKueuZDi2-" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 727} outputId="222fafc5-0ad5-4579-8b5d-74c995e59a3a" executionInfo={"status": "ok", "timestamp": 1588233943531, "user_tz": -330, "elapsed": 2969, "user": {"displayName": "Manmohan Dogra", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhoL7CtYUEHr9zkw97VpbpdDqYV30gXSuOX9CSBdA=s64", "userId": "04603716801366227129"}}
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    #after 6 layers we use flatten to create single vector along with activation function

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),

  #since it's a multi-class hence we'll use softmax activation function.

    tf.keras.layers.Dense(4, activation='softmax')
])

model.summary()
```

```python id="Umv1gul5Di3I" colab_type="code" colab={}
#compiling the model by setting the type of classifier, optimizer, acc we want in output

#using the RMSprop optimization algorithm is preferable to stochastic 
#gradient descent (SGD), because RMSprop automates learning-rate tuning for us. 
model.compile(optimizer = RMSprop(lr=1e-4),
              loss = 'categorical_crossentropy',metrics=['accuracy'])

```

<!-- #region id="oO_R1BVJrnG8" colab_type="text" -->
## 5. Data Preprocessing

Since the data we've can be of different size and pixes, hence we normalize the image before feeding it to the NN.


Training the model with the Augmented data on the way is really a productive way of getting better results. The augmentation trains the model on the various operaton performed on the image to increase the dataset without affecting the size of dataset.
<!-- #endregion -->

```python id="r8yp3-w4siri" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 52} outputId="26162ec6-f743-42b1-b897-ff6ed7256f46" executionInfo={"status": "ok", "timestamp": 1588234194535, "user_tz": -330, "elapsed": 1721, "user": {"displayName": "Manmohan Dogra", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhoL7CtYUEHr9zkw97VpbpdDqYV30gXSuOX9CSBdA=s64", "userId": "04603716801366227129"}}
## Using Augmentations

train_datagen = ImageDataGenerator(
      rescale=1./255.,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255.)

train_generator = train_datagen.flow_from_directory(
        training_dir,  # This is the source directory for training images
        target_size=(200, 200),  # All images will be resized to 200x200
        batch_size=25,
        # Since we use sparse_categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(200, 200),
        batch_size=20,
        class_mode='categorical')
```

<!-- #region id="eGzeMBmEtpI-" colab_type="text" -->
## 6. Training the Model


<!-- #endregion -->

```python id="tNT26AMLDi3Q" colab_type="code" outputId="be5894df-83cd-425d-99d6-78644e922547" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1588238342821, "user_tz": -330, "elapsed": 4143736, "user": {"displayName": "Manmohan Dogra", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhoL7CtYUEHr9zkw97VpbpdDqYV30gXSuOX9CSBdA=s64", "userId": "04603716801366227129"}}


history = model.fit(
      train_generator,
      steps_per_epoch=59,  # 1200 images = batch_size * steps
      epochs=180,
      validation_data=validation_generator,
      validation_steps=30,
     verbose=1
      
) # 1200 images = batch_size * steps)
```

<!-- #region id="xbbA2VprvvcK" colab_type="text" -->
## 7.1 Visualization of the Results.
<!-- #endregion -->

```python id="BiKnsMNCDi3x" colab_type="code" outputId="6100ebe3-0691-4191-aff5-37eafdb037d7" executionInfo={"status": "ok", "timestamp": 1588238557720, "user_tz": -330, "elapsed": 2657, "user": {"displayName": "Manmohan Dogra", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhoL7CtYUEHr9zkw97VpbpdDqYV30gXSuOX9CSBdA=s64", "userId": "04603716801366227129"}} colab={"base_uri": "https://localhost:8080/", "height": 298}
# Plot the chart for accuracy and loss on both training and validation
%matplotlib inline
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()



plt.show()
```

<!-- #region id="HS6K39A3Za23" colab_type="text" -->
Hence, we can say that the model performed well with the accuracy of [ 84% ] approx.
<!-- #endregion -->

<!-- #region id="OYCHn6LDxdH6" colab_type="text" -->

## 8. Testing the Model
Taking pictures from the validation set to test the model
<!-- #endregion -->

```python id="sbUvac0txY68" colab_type="code" colab={"resources": {"http://localhost:8080/nbextensions/google.colab/files.js": {"data": "Ly8gQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQwovLwovLyBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgIkxpY2Vuc2UiKTsKLy8geW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLgovLyBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXQKLy8KLy8gICAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjAKLy8KLy8gVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZQovLyBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiAiQVMgSVMiIEJBU0lTLAovLyBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC4KLy8gU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZAovLyBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS4KCi8qKgogKiBAZmlsZW92ZXJ2aWV3IEhlbHBlcnMgZm9yIGdvb2dsZS5jb2xhYiBQeXRob24gbW9kdWxlLgogKi8KKGZ1bmN0aW9uKHNjb3BlKSB7CmZ1bmN0aW9uIHNwYW4odGV4dCwgc3R5bGVBdHRyaWJ1dGVzID0ge30pIHsKICBjb25zdCBlbGVtZW50ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnc3BhbicpOwogIGVsZW1lbnQudGV4dENvbnRlbnQgPSB0ZXh0OwogIGZvciAoY29uc3Qga2V5IG9mIE9iamVjdC5rZXlzKHN0eWxlQXR0cmlidXRlcykpIHsKICAgIGVsZW1lbnQuc3R5bGVba2V5XSA9IHN0eWxlQXR0cmlidXRlc1trZXldOwogIH0KICByZXR1cm4gZWxlbWVudDsKfQoKLy8gTWF4IG51bWJlciBvZiBieXRlcyB3aGljaCB3aWxsIGJlIHVwbG9hZGVkIGF0IGEgdGltZS4KY29uc3QgTUFYX1BBWUxPQURfU0laRSA9IDEwMCAqIDEwMjQ7Ci8vIE1heCBhbW91bnQgb2YgdGltZSB0byBibG9jayB3YWl0aW5nIGZvciB0aGUgdXNlci4KY29uc3QgRklMRV9DSEFOR0VfVElNRU9VVF9NUyA9IDMwICogMTAwMDsKCmZ1bmN0aW9uIF91cGxvYWRGaWxlcyhpbnB1dElkLCBvdXRwdXRJZCkgewogIGNvbnN0IHN0ZXBzID0gdXBsb2FkRmlsZXNTdGVwKGlucHV0SWQsIG91dHB1dElkKTsKICBjb25zdCBvdXRwdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQob3V0cHV0SWQpOwogIC8vIENhY2hlIHN0ZXBzIG9uIHRoZSBvdXRwdXRFbGVtZW50IHRvIG1ha2UgaXQgYXZhaWxhYmxlIGZvciB0aGUgbmV4dCBjYWxsCiAgLy8gdG8gdXBsb2FkRmlsZXNDb250aW51ZSBmcm9tIFB5dGhvbi4KICBvdXRwdXRFbGVtZW50LnN0ZXBzID0gc3RlcHM7CgogIHJldHVybiBfdXBsb2FkRmlsZXNDb250aW51ZShvdXRwdXRJZCk7Cn0KCi8vIFRoaXMgaXMgcm91Z2hseSBhbiBhc3luYyBnZW5lcmF0b3IgKG5vdCBzdXBwb3J0ZWQgaW4gdGhlIGJyb3dzZXIgeWV0KSwKLy8gd2hlcmUgdGhlcmUgYXJlIG11bHRpcGxlIGFzeW5jaHJvbm91cyBzdGVwcyBhbmQgdGhlIFB5dGhvbiBzaWRlIGlzIGdvaW5nCi8vIHRvIHBvbGwgZm9yIGNvbXBsZXRpb24gb2YgZWFjaCBzdGVwLgovLyBUaGlzIHVzZXMgYSBQcm9taXNlIHRvIGJsb2NrIHRoZSBweXRob24gc2lkZSBvbiBjb21wbGV0aW9uIG9mIGVhY2ggc3RlcCwKLy8gdGhlbiBwYXNzZXMgdGhlIHJlc3VsdCBvZiB0aGUgcHJldmlvdXMgc3RlcCBhcyB0aGUgaW5wdXQgdG8gdGhlIG5leHQgc3RlcC4KZnVuY3Rpb24gX3VwbG9hZEZpbGVzQ29udGludWUob3V0cHV0SWQpIHsKICBjb25zdCBvdXRwdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQob3V0cHV0SWQpOwogIGNvbnN0IHN0ZXBzID0gb3V0cHV0RWxlbWVudC5zdGVwczsKCiAgY29uc3QgbmV4dCA9IHN0ZXBzLm5leHQob3V0cHV0RWxlbWVudC5sYXN0UHJvbWlzZVZhbHVlKTsKICByZXR1cm4gUHJvbWlzZS5yZXNvbHZlKG5leHQudmFsdWUucHJvbWlzZSkudGhlbigodmFsdWUpID0+IHsKICAgIC8vIENhY2hlIHRoZSBsYXN0IHByb21pc2UgdmFsdWUgdG8gbWFrZSBpdCBhdmFpbGFibGUgdG8gdGhlIG5leHQKICAgIC8vIHN0ZXAgb2YgdGhlIGdlbmVyYXRvci4KICAgIG91dHB1dEVsZW1lbnQubGFzdFByb21pc2VWYWx1ZSA9IHZhbHVlOwogICAgcmV0dXJuIG5leHQudmFsdWUucmVzcG9uc2U7CiAgfSk7Cn0KCi8qKgogKiBHZW5lcmF0b3IgZnVuY3Rpb24gd2hpY2ggaXMgY2FsbGVkIGJldHdlZW4gZWFjaCBhc3luYyBzdGVwIG9mIHRoZSB1cGxvYWQKICogcHJvY2Vzcy4KICogQHBhcmFtIHtzdHJpbmd9IGlucHV0SWQgRWxlbWVudCBJRCBvZiB0aGUgaW5wdXQgZmlsZSBwaWNrZXIgZWxlbWVudC4KICogQHBhcmFtIHtzdHJpbmd9IG91dHB1dElkIEVsZW1lbnQgSUQgb2YgdGhlIG91dHB1dCBkaXNwbGF5LgogKiBAcmV0dXJuIHshSXRlcmFibGU8IU9iamVjdD59IEl0ZXJhYmxlIG9mIG5leHQgc3RlcHMuCiAqLwpmdW5jdGlvbiogdXBsb2FkRmlsZXNTdGVwKGlucHV0SWQsIG91dHB1dElkKSB7CiAgY29uc3QgaW5wdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoaW5wdXRJZCk7CiAgaW5wdXRFbGVtZW50LmRpc2FibGVkID0gZmFsc2U7CgogIGNvbnN0IG91dHB1dEVsZW1lbnQgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChvdXRwdXRJZCk7CiAgb3V0cHV0RWxlbWVudC5pbm5lckhUTUwgPSAnJzsKCiAgY29uc3QgcGlja2VkUHJvbWlzZSA9IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7CiAgICBpbnB1dEVsZW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignY2hhbmdlJywgKGUpID0+IHsKICAgICAgcmVzb2x2ZShlLnRhcmdldC5maWxlcyk7CiAgICB9KTsKICB9KTsKCiAgY29uc3QgY2FuY2VsID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnYnV0dG9uJyk7CiAgaW5wdXRFbGVtZW50LnBhcmVudEVsZW1lbnQuYXBwZW5kQ2hpbGQoY2FuY2VsKTsKICBjYW5jZWwudGV4dENvbnRlbnQgPSAnQ2FuY2VsIHVwbG9hZCc7CiAgY29uc3QgY2FuY2VsUHJvbWlzZSA9IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7CiAgICBjYW5jZWwub25jbGljayA9ICgpID0+IHsKICAgICAgcmVzb2x2ZShudWxsKTsKICAgIH07CiAgfSk7CgogIC8vIENhbmNlbCB1cGxvYWQgaWYgdXNlciBoYXNuJ3QgcGlja2VkIGFueXRoaW5nIGluIHRpbWVvdXQuCiAgY29uc3QgdGltZW91dFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgc2V0VGltZW91dCgoKSA9PiB7CiAgICAgIHJlc29sdmUobnVsbCk7CiAgICB9LCBGSUxFX0NIQU5HRV9USU1FT1VUX01TKTsKICB9KTsKCiAgLy8gV2FpdCBmb3IgdGhlIHVzZXIgdG8gcGljayB0aGUgZmlsZXMuCiAgY29uc3QgZmlsZXMgPSB5aWVsZCB7CiAgICBwcm9taXNlOiBQcm9taXNlLnJhY2UoW3BpY2tlZFByb21pc2UsIHRpbWVvdXRQcm9taXNlLCBjYW5jZWxQcm9taXNlXSksCiAgICByZXNwb25zZTogewogICAgICBhY3Rpb246ICdzdGFydGluZycsCiAgICB9CiAgfTsKCiAgaWYgKCFmaWxlcykgewogICAgcmV0dXJuIHsKICAgICAgcmVzcG9uc2U6IHsKICAgICAgICBhY3Rpb246ICdjb21wbGV0ZScsCiAgICAgIH0KICAgIH07CiAgfQoKICBjYW5jZWwucmVtb3ZlKCk7CgogIC8vIERpc2FibGUgdGhlIGlucHV0IGVsZW1lbnQgc2luY2UgZnVydGhlciBwaWNrcyBhcmUgbm90IGFsbG93ZWQuCiAgaW5wdXRFbGVtZW50LmRpc2FibGVkID0gdHJ1ZTsKCiAgZm9yIChjb25zdCBmaWxlIG9mIGZpbGVzKSB7CiAgICBjb25zdCBsaSA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2xpJyk7CiAgICBsaS5hcHBlbmQoc3BhbihmaWxlLm5hbWUsIHtmb250V2VpZ2h0OiAnYm9sZCd9KSk7CiAgICBsaS5hcHBlbmQoc3BhbigKICAgICAgICBgKCR7ZmlsZS50eXBlIHx8ICduL2EnfSkgLSAke2ZpbGUuc2l6ZX0gYnl0ZXMsIGAgKwogICAgICAgIGBsYXN0IG1vZGlmaWVkOiAkewogICAgICAgICAgICBmaWxlLmxhc3RNb2RpZmllZERhdGUgPyBmaWxlLmxhc3RNb2RpZmllZERhdGUudG9Mb2NhbGVEYXRlU3RyaW5nKCkgOgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAnbi9hJ30gLSBgKSk7CiAgICBjb25zdCBwZXJjZW50ID0gc3BhbignMCUgZG9uZScpOwogICAgbGkuYXBwZW5kQ2hpbGQocGVyY2VudCk7CgogICAgb3V0cHV0RWxlbWVudC5hcHBlbmRDaGlsZChsaSk7CgogICAgY29uc3QgZmlsZURhdGFQcm9taXNlID0gbmV3IFByb21pc2UoKHJlc29sdmUpID0+IHsKICAgICAgY29uc3QgcmVhZGVyID0gbmV3IEZpbGVSZWFkZXIoKTsKICAgICAgcmVhZGVyLm9ubG9hZCA9IChlKSA9PiB7CiAgICAgICAgcmVzb2x2ZShlLnRhcmdldC5yZXN1bHQpOwogICAgICB9OwogICAgICByZWFkZXIucmVhZEFzQXJyYXlCdWZmZXIoZmlsZSk7CiAgICB9KTsKICAgIC8vIFdhaXQgZm9yIHRoZSBkYXRhIHRvIGJlIHJlYWR5LgogICAgbGV0IGZpbGVEYXRhID0geWllbGQgewogICAgICBwcm9taXNlOiBmaWxlRGF0YVByb21pc2UsCiAgICAgIHJlc3BvbnNlOiB7CiAgICAgICAgYWN0aW9uOiAnY29udGludWUnLAogICAgICB9CiAgICB9OwoKICAgIC8vIFVzZSBhIGNodW5rZWQgc2VuZGluZyB0byBhdm9pZCBtZXNzYWdlIHNpemUgbGltaXRzLiBTZWUgYi82MjExNTY2MC4KICAgIGxldCBwb3NpdGlvbiA9IDA7CiAgICB3aGlsZSAocG9zaXRpb24gPCBmaWxlRGF0YS5ieXRlTGVuZ3RoKSB7CiAgICAgIGNvbnN0IGxlbmd0aCA9IE1hdGgubWluKGZpbGVEYXRhLmJ5dGVMZW5ndGggLSBwb3NpdGlvbiwgTUFYX1BBWUxPQURfU0laRSk7CiAgICAgIGNvbnN0IGNodW5rID0gbmV3IFVpbnQ4QXJyYXkoZmlsZURhdGEsIHBvc2l0aW9uLCBsZW5ndGgpOwogICAgICBwb3NpdGlvbiArPSBsZW5ndGg7CgogICAgICBjb25zdCBiYXNlNjQgPSBidG9hKFN0cmluZy5mcm9tQ2hhckNvZGUuYXBwbHkobnVsbCwgY2h1bmspKTsKICAgICAgeWllbGQgewogICAgICAgIHJlc3BvbnNlOiB7CiAgICAgICAgICBhY3Rpb246ICdhcHBlbmQnLAogICAgICAgICAgZmlsZTogZmlsZS5uYW1lLAogICAgICAgICAgZGF0YTogYmFzZTY0LAogICAgICAgIH0sCiAgICAgIH07CiAgICAgIHBlcmNlbnQudGV4dENvbnRlbnQgPQogICAgICAgICAgYCR7TWF0aC5yb3VuZCgocG9zaXRpb24gLyBmaWxlRGF0YS5ieXRlTGVuZ3RoKSAqIDEwMCl9JSBkb25lYDsKICAgIH0KICB9CgogIC8vIEFsbCBkb25lLgogIHlpZWxkIHsKICAgIHJlc3BvbnNlOiB7CiAgICAgIGFjdGlvbjogJ2NvbXBsZXRlJywKICAgIH0KICB9Owp9CgpzY29wZS5nb29nbGUgPSBzY29wZS5nb29nbGUgfHwge307CnNjb3BlLmdvb2dsZS5jb2xhYiA9IHNjb3BlLmdvb2dsZS5jb2xhYiB8fCB7fTsKc2NvcGUuZ29vZ2xlLmNvbGFiLl9maWxlcyA9IHsKICBfdXBsb2FkRmlsZXMsCiAgX3VwbG9hZEZpbGVzQ29udGludWUsCn07Cn0pKHNlbGYpOwo=", "ok": true, "headers": [["content-type", "application/javascript"]], "status": 200, "status_text": ""}}, "base_uri": "https://localhost:8080/", "height": 1000} outputId="c42be379-3297-4bb5-a56a-655fbfaf3a8b" executionInfo={"status": "ok", "timestamp": 1588239402578, "user_tz": -330, "elapsed": 41056, "user": {"displayName": "Manmohan Dogra", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhoL7CtYUEHr9zkw97VpbpdDqYV30gXSuOX9CSBdA=s64", "userId": "04603716801366227129"}}
import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
 
  # predicting images
  path = fn
  img = image.load_img(path, target_size=(200, 200))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

 
  plt.imshow(img)

  plt.show()

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(fn)
  print(classes)

  for i in classes:
    if classes[0][0]==1:
      print('It is an Abyssian Cat')
    elif classes[0][1]==1:
      print('It is a Munchkin Cat')
    elif classes[0][2]==1:
      print('It is a Persian Cat')
    elif classes[0][3]==1:
      print('It is a Toyger Cat')
```

<!-- #region id="ikLGxuYDI7B3" colab_type="text" -->
## 9.1 Saving the model
<!-- #endregion -->

```python id="sqH3P7N_I6hg" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="031da9b2-d595-48c8-8c86-c231318e2006" executionInfo={"status": "ok", "timestamp": 1588238363241, "user_tz": -330, "elapsed": 1538, "user": {"displayName": "Manmohan Dogra", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhoL7CtYUEHr9zkw97VpbpdDqYV30gXSuOX9CSBdA=s64", "userId": "04603716801366227129"}}
 
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 

 

 

```

<!-- #region id="JKNW_WeSKOd-" colab_type="text" -->
## 9.2 Load saved model (later) 
<!-- #endregion -->

```python id="9EYNTmVJKzQs" colab_type="code" colab={}
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
```

<!-- #region id="9NODLDSpX3BF" colab_type="text" -->
## 10. Conclusion 
Finally we can say that the model performed really well and was able to predict each breed very well using the CNN with.
<!-- #endregion -->

```python id="nxtnRiIRLSBg" colab_type="code" colab={}

```
