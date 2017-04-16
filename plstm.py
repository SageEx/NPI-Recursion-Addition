'''Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
#!/usr/bin/python
import os
import glob
import subprocess
import json
import re
# from __future__ import print_function
import numpy as np  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from keras.layers import Dense, Dropout, Activation, Embedding , Merge,RepeatVector
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed
from keras.datasets import imdb
from keras.layers.core import Reshape
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
seed=7


np.random.seed(1337)


def RLSTM():
        xtrain=[]
        ytrain=[]
        xtest=[]
        ytest=[]

        X_train = np.loadtxt('inpPlstm.txt')
        Y_train = np.loadtxt('outPlstm.txt')
        Y_train = Y_train.reshape(99, 3, 1)
        X_test = X_train
        Y_test = Y_train
        X_train = np.repeat(X_train, 500, axis = 0)
        Y_train = np.repeat(Y_train, 500, axis = 0)

        print(X_train.shape)
        print(Y_train.shape)

        model = Sequential()
        model.add(Dense(1,input_shape=(1,)))
        print(model.output_shape)
        model.add(RepeatVector(3))
        print(model.output_shape)
        model.add(LSTM(3, return_sequences=True))
        print(model.output_shape)
        # model.add(Bidirectional(LSTM(64)))
        model.add(TimeDistributed(Dense(1, activation='sigmoid')))
        print(model.output_shape)
        # model.add(TimeDistributed(Dense(1, activation='sigmoid')))
        # model.add(Activation('softmax'))

        model.compile('adam', 'mean_squared_error', metrics=['accuracy'])
        model.summary()

        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # model.compile(loss='mean_squared_error', optimizer='sgd',class_mode="binary")
        # model.add(Dense(1, activation='sigmoid')
        print('Train...')
        model.fit(X_train, Y_train, batch_size=2, epochs=130)
        # import ipdb; ipdb.set_trace()
        score = model.evaluate(X_test, Y_test, verbose=0)
        model.save('rLstm.h5')
        Y_result=model.predict(X_test,batch_size=2,verbose=0)
        for num in range(0,len(Y_result)):
            for s in range(0,len(Y_result[num])):
                for u in range(0, len(Y_result[num][s])):
                    print( int(round(Y_result[num][s][u])) )
                # print("\n")
        # print Y_result
        print(score)

RLSTM()



