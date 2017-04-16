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


def ALSTM():
	xtrain=[]
	ytrain=[]
	xtest=[]
	ytest=[]
	# fp1=open("/home/ankit/Desktop/sem8/dl/inpPlstm.txt","r")
	# fp2=open("/home/ankit/Desktop/sem8/dl/outPlstm.txt","r")
	# fp3=open("/home/ankit/Desktop/sem8/dl/inptestPlstm.txt","r")
	# fp4=open("/home/ankit/Desktop/sem8/dl/outtestPlstm.txt","r")
	# for line in fp1:
	# 	line=line.rstrip()
	# 	if len(line)!=0:
	# 		xtrain.append(line)
	# for line in fp2:
	# 	line=line.rstrip()
	# 	if len(line)!=0:
	# 		line=line.split(",")
	# 		ytrain.append(line)
	# for line in fp3:
	# 	line=line.rstrip()
	# 	if len(line)!=0:
	# 		xtest.append(line)
	# for line in fp4:
	# 	line=line.rstrip()
	# 	if len(line)!=0:
	# 		line=line.split(",")
	# 		ytest.append(line)
	# X_train = np.array(xtrain, np.float32)
	# Y_train = np.array(ytrain, np.float32)
	# X_test = np.array(xtest, np.float32)
	# Y_test = np.array(ytest, np.float32)
	X_train = np.loadtxt('/home/ankit/Desktop/sem8/dl/inpAlstm.txt')
	Y_train = np.loadtxt('/home/ankit/Desktop/sem8/dl/outAlstm.txt')
	X_test = X_train
	Y_test = Y_train
	print X_train.shape
	print Y_train.shape
	model = Sequential()
	model.add(Reshape((1,4),input_shape=(4,)))
	# model.add(Dense(3))
	model.add((LSTM(3)))
	model.add(Dense(4))
	# model.add(Activation('softmax'))
	model.compile('adadelta', 'mean_squared_error', metrics=['accuracy'])

	# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	# model.compile(loss='mean_squared_error', optimizer='sgd',class_mode="binary")
	# model.add(Dense(1, activation='sigmoid')
	print('Train...')
	model.fit(X_train, Y_train, batch_size=2, nb_epoch=130)
	# import ipdb; ipdb.set_trace()
	score = model.evaluate(X_test, Y_test, verbose=0)
	model_json = model.to_json()
	with open("Amodel.json", "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("Amodel.h5")
	print("Saved model to disk")
	# model.save('rLstm.h5')
	Y_result=model.predict(X_test,batch_size=2,verbose=0)
	for num in range(0,len(Y_result)):
		for s in range(0,len(Y_result[num])):
			print int(round(Y_result[num][s]))
		print "\n"
	# print Y_result
	print score

ALSTM()