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