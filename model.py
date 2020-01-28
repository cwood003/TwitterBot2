from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

# building the model

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=())
])