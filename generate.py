import sys


def makeInputs():
    ret = []
    for i in range(0, 10):
        for j in range(0, 10):
            st = str(i) + "\t" + str(j) + "\t"
            st += str((i+j)%10) + "\t" + str(int((i+j)/10))
            print(st)
            ret.append(st)

    return ret

def writeToFile(fi, inp):
    with open(fi, 'w') as f:
        for st in inp:
            f.write(st)
            f.write("\n")


if __name__ == '__main__':
    fi = sys.argv[1]
    inp = makeInputs()
    print(inp[0])
    writeToFile(fi, inp)
