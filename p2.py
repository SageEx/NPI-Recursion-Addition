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
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

json_file = open('Rmodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model1 = model_from_json(loaded_model_json)
loaded_model1.load_weights("Rmodel.h5")
json_file = open('Amodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model2 = model_from_json(loaded_model_json)
# load weights into new model
loaded_model2.load_weights("Amodel.h5")
json_file = open('Pmodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Pmodel.h5")


def Rlstm(b):
	X_test=[]
	X_test.append(b)
	X_test.append(b)
	
	# load weights into new model
	# print("Loaded model from disk")
	 
	# evaluate loaded model on test data
	loaded_model1.compile('adadelta', 'mean_squared_error', metrics=['accuracy'])
	Y_result=loaded_model1.predict(X_test,batch_size=2,verbose=0)
	for num in range(0,len(Y_result)):
		for s in range(0,len(Y_result[num])):
			Y_result[num][s] =int(round(Y_result[num][s]))
	out=map(int,Y_result[0].tolist())
	return out

def Alstm(b):
	X_test=[]
	X_test.append(b)
	X_test.append(b)
	
	# print("Loaded model from disk")
	 
	# evaluate loaded model on test data
	loaded_model2.compile('adadelta', 'mean_squared_error', metrics=['accuracy'])
	Y_result=loaded_model2.predict(X_test,batch_size=2,verbose=0)
	for num in range(0,len(Y_result)):
		for s in range(0,len(Y_result[num])):
			Y_result[num][s] =int(round(Y_result[num][s]))
	out=map(int,Y_result[0].tolist())
	return out

def Plstm(b):
	X=[]
	X.append(b)
	X_test=[]
	X_test.append(X)
	X_test.append(X)
	# print X_test
	
	# print("Loaded model from disk")
	 
	# evaluate loaded model on test data
	loaded_model3.compile('adadelta', 'mean_squared_error', metrics=['accuracy'])
	Y_result=loaded_model3.predict(X_test,batch_size=2,verbose=0)
	for num in range(0,len(Y_result)):
		for s in range(0,len(Y_result[num])):
			Y_result[num][s] =int(round(Y_result[num][s]))
	out=map(int,Y_result[0].tolist())
	return out

a=Rlstm([1,2,3,1])
print a
a=Rlstm([2,4,5,1])
print a
a=Rlstm([3,6,6,6])
print a

