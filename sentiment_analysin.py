import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Load dataset
vocab_size = 10000  # top 10k words
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Padding (make same length)
max_len = 200

x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# Build Model
model = Sequential()

model.add(Embedding(vocab_size, 128, input_length=max_len))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Summary
model.summary()

# Train
model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# Test
loss, acc = model.evaluate(x_test, y_test)

print("Test Accuracy:", acc)

# Save Model
model.save("sentiment_lstm_model.h5")