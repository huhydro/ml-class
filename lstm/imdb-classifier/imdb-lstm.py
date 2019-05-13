from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import CuDNNLSTM as LSTM
from keras.layers import Conv1D, Flatten
from keras.layers import Bidirectional
from keras.datasets import imdb
import wandb
from wandb.keras import WandbCallback
import imdb
import numpy as np
from keras.preprocessing import text

wandb.init()
config = wandb.config

# set parameters:
config.vocab_size = 2000
config.maxlen = 300
config.batch_size = 32
config.embedding_dims = 50
config.filters = 250
config.kernel_size = 3
config.hidden_dims = 100
config.epochs = 10

(X_train, y_train), (X_test, y_test) = imdb.load_imdb()

print("Before ", X_train[0])
tokenizer = text.Tokenizer(num_words=config.vocab_size)
tokenizer.fit_on_texts(X_train)
print("Mid ", X_train[0])

X_train = tokenizer.texts_to_sequences(X_train)
print("After ", X_train[0])
#print("X_train.shape",X_train.shape)
#exit()
X_test = tokenizer.texts_to_sequences(X_test)

X_train = sequence.pad_sequences(X_train, maxlen=config.maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=config.maxlen)

model = Sequential()
model.add(Embedding(config.vocab_size,
                    config.embedding_dims,
                    input_length=config.maxlen))
model.add(Bidirectional(LSTM(150)))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=config.batch_size,
          epochs=config.epochs,
          validation_data=(X_test, y_test), callbacks=[WandbCallback()])

model.save("seniment.h5")
