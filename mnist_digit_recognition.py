
# Handwritten Digit Recognition (MNIST) using TensorFlow/Keras

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

print("Training data shape:", x_train.shape)

# Build the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train_cat, epochs=5, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print(f"Test accuracy: {test_acc:.2f}")

# Predict one sample
sample_index = 100
plt.imshow(x_test[sample_index], cmap='gray')
plt.title("Actual Label: " + str(y_test[sample_index]))
plt.show()

prediction = model.predict(x_test[sample_index].reshape(1, 28, 28))
print("Predicted Digit:", np.argmax(prediction))

# Add Gradio interface
import gradio as gr

def recognize_digit(image):
    image = image.reshape(1, 28, 28)
    pred = model.predict(image).argmax()
    return f"Predicted Digit: {pred}"

gr.Interface(fn=recognize_digit, inputs="sketchpad", outputs="text", live=True).launch()

# Save the model
model.save("mnist_digit_model.h5")
