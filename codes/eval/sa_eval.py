# coding:utf-8
from __future__ import division
import sys
goldDict1 = {}
goldDict2 = {}
labelDict1 = {}
labelDict2 = {}
tp = 0
fp = 0
fn1 = 0
fn2 = 0
codes = []
def compare(data, gold, label):
    global goldDict1, goldDict2, labelDict1, labelDict2
    global tp, fp, fn1, fn2
    global codes
    for line in data:
        temp = line.strip().split('\t')
        codes += [temp[0]]
    for line in gold:
        temp = line.strip().split('\t')
        if temp[0] not in goldDict1:
            goldDict1[temp[0]] = [temp[1]]
            goldDict2[temp[0]] = [temp[2]]
        else:
            goldDict1[temp[0]] += [temp[1]]
            goldDict2[temp[0]] += [temp[2]]
    for line in label:
        temp = line.strip().split('\t')
        if temp[0] not in labelDict1:
            labelDict1[temp[0]] = [temp[1]]
            labelDict2[temp[0]] = [temp[2]]
        else:
            labelDict1[temp[0]] += [temp[1]]
            labelDict2[temp[0]] += [temp[2]]

    for code in codes:
        if code not in goldDict1 and code not in labelDict1:
            continue
        elif code not in goldDict1 and code in labelDict1:
            fn2 += len(labelDict1[code])
        elif code not in labelDict1 and code in goldDict1:
            fn1 += len(goldDict1[code])
        else:
            i = 0
            while i < len(labelDict1[code]):
                aspect = labelDict1[code][i]
                if aspect not in goldDict1[code]:
                    fn2 += 1
                else:
                    if labelDict2[code][i] == goldDict2[code][goldDict1[code].index(aspect)]:
                        tp += 1
                    else:
                        fp += 1
                i += 1
            
            i = 0
            while i < len(goldDict1[code]):
                aspect = goldDict1[code][i]
                if aspect not in labelDict1[code]:
                    fn1 += 1
                i += 1
    
    P = tp / (tp + fp + fn2)
    R = tp / (tp + fn1)
    if P+R != 0:
        F1 = 2*P*R/(P+R)
        return [F1, P, R]
    else:
        return [0.0, P, R]

if __name__ == '__main__':
    filename = sys.argv[1]
    dataFile = open('../../data/'+filename+'.csv', 'r')
    goldFile = open('../../data/Label_Gold_'+filename+'.csv', 'r') 
    labelFile = open('../../data/detect/Label_'+filename+'.csv', 'r')
    dataFile.next()
    goldFile.next()
    labelFile.next()
    F1, P, R = compare(dataFile, goldFile, labelFile)
    print 'F1:\t%f'%F1
    print 'P:\t%f'%P
    print 'R:\t%f'%R
