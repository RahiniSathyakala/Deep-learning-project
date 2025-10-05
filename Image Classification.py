# Deep Learning Model for Fashion-MNIST Image Classification
# Using TensorFlow & Keras

import tensorflow as tf
from tensorflow.keras import datasets, models, layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random

# ------------------------------
# 1. Load and Preprocess Dataset
# ------------------------------
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

# Normalize (0-255 -> 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN input (28x28 images -> add channel dimension)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Class labels
class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# ------------------------------
# 2. Build CNN Model
# ------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Show model summary
model.summary()

# ------------------------------
# 3. Train the Model
# ------------------------------
history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))

# ------------------------------
# 4. Visualize Accuracy & Loss
# ------------------------------
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# ------------------------------
# 5. Evaluate Model on Test Data
# ------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("\nTest Accuracy: {:.2f}%".format(test_acc * 100))

# ------------------------------
# 6. Confusion Matrix
# ------------------------------
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

# ------------------------------
# 7. Sample Predictions
# ------------------------------
plt.figure(figsize=(10,10))
for i in range(9):
    idx = random.randint(0, len(x_test)-1)
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[idx].reshape(28,28), cmap='gray')
    plt.title(f"True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred_classes[idx]]}")
    plt.axis('off')
plt.show()

