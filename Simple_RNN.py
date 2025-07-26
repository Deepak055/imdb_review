import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,SimpleRNN  
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence


max_features = 10000  # Number of unique words to consider

(X_train, y_train),(X_test,y_test) = imdb.load_data(num_words=max_features, maxlen=max_features)

# print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
# print(f"Test data shape: {X_test.shape}, Labels shape: {y_test.shape}")

# print(X_train[0])  # Print the first training sample
sample_review=X_train[0]
sample_label=y_train[0]
# print(f"Sample review: {sample_review}, Label: {sample_label}")

word_index=imdb.get_word_index()
reversed_word_index = {v: k for k, v in word_index.items()}
# print("Sample review words:", [reversed_word_index.get(i-3, '?') for i in sample_review])

#sequence
X_train = sequence.pad_sequences(X_train, maxlen=500)
X_test = sequence.pad_sequences(X_test, maxlen=500)  
  
# print(f"Shape of training data after padding: {X_train}")

# Build the model
model = Sequential()
# The input size: each sentence (or review) is padded or truncated to 500 words.
model.add(Embedding(max_features, 128, input_length=500))  # Embedding layer
# The dimension of the embedding vector. Each word will be converted into a 128-dimensional vector.
model.add(SimpleRNN(128, activation='relu'))  # Simple RNN layer

# The number of RNN units (output dimensions)
model.add(Dense(1, activation='sigmoid'))  # Dense layer
model.summary()  # Print model summary

# Compile the model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#callbaack
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
# The model will go through all 1,000 samples 10 times, adjusting its weights each time
history=model.fit(X_train, y_train, epochs=10,batch_size=32, validation_data=(X_test,y_test), callbacks=[early_stopping])
# accuracy: 0.8437 - loss: 0.3636 - val_accuracy: 0.7931 - val_loss: 0.4474
model.save('imdb.h5')