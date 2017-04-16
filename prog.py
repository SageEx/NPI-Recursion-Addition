from __future__ import print_function
#from seq2seq import AttentionSeq2Seq

import numpy as np
import pickle
import sys

import keras
from keras import backend as K
from keras.utils import generic_utils
from keras.utils.np_utils import to_categorical
# from keras.preprocessing import sequence
# from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.layers import Bidirectional, Reshape, Activation, TimeDistributed
from keras.layers.normalization import BatchNormalization

CLASS_FILE = sys.argv[1]
MODEL_FILE = CLASS_FILE + '_model.pkl'

np.random.seed(1337)  # for reproducibility
# NUM_THREADS = 16
# sess = tf.Session(config=tf.ConfigProto(
#     intra_op_parallelism_threads=NUM_THREADS))
# K.set_session(sess)

TRAIN_SIZE=70
TEST_SIZE=30
# TRAIN_SIZE=3045
# TEST_SIZE=800
# SENT_SIZE=85

# Model
model = Sequential()
model.add(Dense(3, input_shape=(3,)))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('relu'))
# model.add(Activation('softmax'))

model.compile(loss='mse', optimizer='adam',
              metrics=['accuracy'])
model.summary()

# earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=200,
                                              # verbose=0, mode='auto')

modelCheck = keras.callbacks.ModelCheckpoint(MODEL_FILE, monitor='val_loss',
                                              verbose=0, save_best_only=False, mode='auto')


print('Loading data...')

train_mat = np.loadtxt('inp.txt')
train_mat = np.repeat(train_mat, 500, axis=0)
test_mat = np.loadtxt('inp.txt')

X_train = train_mat[:,:3]
X_test = test_mat[:,:3]
# X_train0 = train_mat[:,0]
# X_train1 = train_mat[:,1]
# X_train0 = to_categorical(X_train0, num_classes=10)
# X_train1 = to_categorical(X_train1, num_classes=10)
# X_train = np.concatenate((X_train0,X_train1), axis=1)
# X_test = X_train
# X_train = np.repeat(X_train, 500, axis=0)

y_train = train_mat[:,3]
# y_train = to_categorical(y_train, num_classes=10)
# y_train = np.repeat(y_train, 500, axis=0)
y_test = test_mat[:,3]
# y_test = to_categorical(y_test, num_classes=10)

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


nb_Epoch=2000
nb_batch=1
print('Train...')


model.fit(X_train, y_train,
          epochs=nb_Epoch,
          batch_size=10,
          shuffle=True,
          callbacks=[ modelCheck],
          validation_data=[X_test, y_test])

del model

bmodel = keras.models.load_model(MODEL_FILE)

score = bmodel.evaluate(X_test, y_test, verbose=0)

Y_result_class = bmodel.predict_classes(X_test, batch_size=1)
np.savetxt(CLASS_FILE, Y_result_class)

import ipdb; ipdb.set_trace()
#layerop = np.array(layeropFn([X_test, 0])).reshape(1083,12800)
#np.savetxt("INT_TEST.out", layerop)
#layerop = np.array(layeropFn([X_train, 0])).reshape(4334, 12800)
#np.savetxt("INT_TRAIN.out", layerop)
