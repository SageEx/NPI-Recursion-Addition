#!/usr/bin/python
import os
import glob
import subprocess
import json
import re
import sys

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
loaded_model2.load_weights("Amodel.h5")
json_file = open('Pmodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model3 = model_from_json(loaded_model_json)
loaded_model3.load_weights("Pmodel.h5")


x = {}
x[1] = [8,5,9,9,5]
x[2] = [0,5,6,8,7]
assert(len(x[1]) == len(x[2]))
x[3] = [0] * (len(x[1]) + 1)

i = {}
i[1] = len(x[1]) -1
i[2] = len(x[2]) -1
i[3] = len(x[3]) -1

carryFlag = 0

def call(pid, aid):
    if pid == 4:
        addx()
    elif pid == 5:
        carry()
    elif pid == 6:
        ptrShift(aid)


def addx():
    print("ADDX\n")
    global carryFlag
    out = (x[1][i[1]] + x[2][i[2]] + carryFlag) %10
    x[3][i[3]] = out


def carry():
    print("CARRY\n")
    global carryFlag
    c = int((x[1][i[1]] + x[2][i[2]] + carryFlag)/10)
    carryFlag = c


def ptrShift(id):
    print("PTR: " + str(id) + "\n")
    i[id] -= 1


def isPrimitive(id):
    return id > 3


def getPrograms(pid):
    progs = Plstm(pid)
    progs.insert(0,pid)
    return progs
    # if pid == 1:
        # return [1,2,3,1]
    # elif pid == 2:
        # return [2,4,5,1]
    # elif pid == 3:
        # return [3,6,6,6]
    # else:
        # raise ValueError("Bullshit Input .. ??")


def getArguments(programs):
    return Alstm(programs)
    # i = programs[0]
    # if i == 1:
        # return [0,0,0,0]
    # elif i == 2:
        # return [0,0,0,0]
    # elif i == 3:
        # return [0,1,2,3]
    # else:
        # raise ValueError("Bullshit Input .. ??")


def getProbabilities(programs):
    return Rlstm(programs)
    # i = programs[0]
    # if i == 1:
        # return [0,0,0,1]
    # elif i == 2:
        # return [0,0,1,1]
    # elif i == 3:
        # return [0,0,0,0]
    # else:
        # raise ValueError("Bullshit Input .. ??")


def Rlstm(b):
        X_test=[]
        X_test.append(b)
        X_test.append(b)

        # load weights into new model
        # print("Loaded model from disk")

        # evaluate loaded model on test data
        Y_result=loaded_model1.predict(X_test,batch_size=2,verbose=0)
        for num in range(0,len(Y_result)):
            for s in range(0,len(Y_result[num])):
                Y_result[num][s] =int(round(Y_result[num][s]))
        out=list(map(int,Y_result[0].tolist()))
        return out


def Alstm(b):
        X_test=[]
        X_test.append(b)
        X_test.append(b)

        # print("Loaded model from disk")

        # evaluate loaded model on test data
        Y_result=loaded_model2.predict(X_test,batch_size=2,verbose=0)
        for num in range(0,len(Y_result)):
            for s in range(0,len(Y_result[num])):
                Y_result[num][s] =int(round(Y_result[num][s]))
        out=list(map(int,Y_result[0].tolist()))
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
    Y_result=loaded_model3.predict(X_test,batch_size=2,verbose=0)
    for num in range(0,len(Y_result)):
        for s in range(0,len(Y_result[num])):
            Y_result[num][s] =int(round(Y_result[num][s]))
    out=list(map(int,Y_result[0].tolist()))
    return out


def RUN(p,a):
    if i[1] < 0:
        global carryFlag
        x[3][i[3]] = carryFlag
        return
    # print(str(i))
    print("RUN:" + str(p) + " "+ str(a) + "\n")
    programs = getPrograms(p)
    probabilities = getProbabilities(programs)
    arguments = getArguments(programs)

    programs.pop(0)
    arguments.pop(0)
    probabilities.pop(0)

    r = 0
    for pid, aid, prid in zip(programs, arguments, probabilities):
        if isPrimitive(pid):
            call(pid, aid)
        else:
            RUN(pid, aid)

        if prid > 0.5:
            break


if __name__ == '__main__':
    # import ipdb; ipdb.set_trace()
    RUN(1,0)
    print(x[1])
    print(x[2])
    print(x[3])
