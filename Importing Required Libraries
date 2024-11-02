
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import tensorflow.keras as keras
import seaborn as sns
import warnings
from textwrap import wrap
import keras.models as models
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, 
img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense,Reshape,Concatenate, 
concatenate, Bidirectional,add
from tensorflow.keras.layers import Conv2D, MaxPooling2D, 
GlobalAveragePooling2D, Activation, Dropout, Flatten, Dense, Input, Layer
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import LeakyReLU
text_path = "/content/flickr8k/captions.txt"
data = pd.read_csv('/content/flickr8k/captions.txt', header = None)
print(data)
