# import CNN tools
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.datasets import fashion_mnist
from keras.layers.core import Dense, Dropout, Activation # Types of layers to be used in our model

from sklearn.model_selection import train_test_split # some helper from scikit for data split
from keras.models import Sequential  # Model type to be used

%matplotlib inline
import matplotlib.pyplot as plt      # MATLAB like plotting routines
import random                        # for generating random numbers

from keras.utils import np_utils                         # NumPy related tools

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Reload the MNIST data to get the original shape (not the flattened 1D vector)
(X_train,y_train), (X_test,y_test) = fashion_mnist.load_data()

# Split the train and a majority validation set to emphasize data dependence (same as before)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.9, random_state=42)

print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_val shape", X_val.shape)
print("y_val shape", y_val.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

# Again, do some formatting
# Except don't flatten images into a 784-length vector so that 2D convolutions can first be performed

X_train = X_train.astype('float32')         # change integers to 32-bit floating point numbers
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255                              # normalize each value for each pixel for the entire vector for each input
X_val /= 255
X_test /= 255

X_train = X_train.reshape(-1,28,28,1)
X_val = X_val.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

print("Training matrix shape", X_train.shape)
print("Validation matrix shape", X_val.shape)
print("Testing matrix shape", X_test.shape)

#Visualisation
class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize=(10, 10))
for i in range(36):
    plt.subplot(6, 6, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i].reshape((28,28)))
    label_index = int(y_train[i])
    plt.title(class_names[label_index])
plt.show()

#One-hot encoding
# one-hot format classes

nb_classes = 10 # number of unique digits

y_train = np_utils.to_categorical(y_train, nb_classes)
y_val = np_utils.to_categorical(y_val, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

#Layers
model = Sequential()                                 # Linear stacking of layers

# Convolution Layer 1
model.add(Conv2D(32, (3, 3), input_shape=(28,28,1))) # 32 different 3x3 kernels -- so 32 feature maps
model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
model.add(Activation('relu'))                        # activation
# model.add(Dropout(0.2))                              # 20% dropout of randomly selected nodes

# Convolution Layer 2
model.add(Conv2D(32, (3, 3)))                        # 32 different 3x3 kernels -- so 32 feature maps
model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
model.add(Activation('relu'))                        # activation
model.add(MaxPooling2D(pool_size=(2,2)))         # Pool the max values over a 2x2 kernel
model.add(Dropout(0.2))                              # 20% dropout of randomly selected nodes

# Convolution Layer 3
model.add(Conv2D(64,(3, 3)))                         # 64 different 3x3 kernels -- so 64 feature maps
model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
model.add(Activation('relu'))                     # activation
# model.add(Dropout(0.2))                              # 20% dropout of randomly selected nodes

# Convolution Layer 4
model.add(Conv2D(64, (3, 3)))                        # 64 different 3x3 kernels -- so 64 feature maps
model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
model.add(Activation('relu'))                        # activation
model.add(MaxPooling2D(pool_size=(2,2)))          # Pool the max values over a 2x2 kernel
model.add(Flatten())                                 # Flatten final 4x4x64 output matrix into a 1024-length vector
# model.add(Dropout(0.2))                              # 20% dropout of randomly selected nodes

# Fully Connected Layer 5
model.add(Dense(512))                                # 512 FCN nodes
model.add(BatchNormalization())                      # normalization
model.add(Activation('relu'))                        # activation
# model.add(Dropout(0.2))                              # 20% dropout of randomly selected nodes

# Fully Connected Layer 6
model.add(Dense(10))                                 # final 10 FCN nodes
model.add(Activation('softmax'))                     # softmax activation

model.summary()

# we'll use the same optimizer

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Besides loss function considerations as before, this method actually results in significant memory savings
# because we are actually LOADING the data into the network in batches before processing each batch
batch_sz = 128
num_epochs = 100

model.fit(X_train, y_train, batch_size=batch_sz, epochs=num_epochs, validation_data=(X_val, y_val), verbose=1)