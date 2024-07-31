import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical

# Loading the data
(train_imgs, train_labels), (test_imgs, test_labels) = cifar100.load_data()

# Normalizing pixel values between 0 and 1
train_imgs = train_imgs / 255.0
test_imgs = test_imgs / 255.0

# One-hot encoding the labels
train_labels = to_categorical(train_labels, num_classes=100)
test_labels = to_categorical(test_labels, num_classes=100)

base_model = MobileNetV2(input_shape=(32, 32, 3), weights='imagenet', include_top=False)

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(100, activation='softmax')
])

model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(
    train_imgs, train_labels,
    validation_data=(test_imgs, test_labels),
    epochs=10,
    batch_size=32
)

model.save('cifar100_classification_preset_model.keras')