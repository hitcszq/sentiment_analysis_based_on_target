# coding:utf-8

from __future__ import division
import json

#评估评价对象抽取结果的精准率和召回率
label = {}
with open('../../data/Label.csv', 'r') as f:
    f.next()
    for line in f:
        temp = line.strip().split('\t')
        if temp[0] not in label:
            label[temp[0].decode('utf-8')] = [temp[1].decode('utf-8')]
        else:
            label[temp[0].decode('utf-8')] += [temp[1].decode('utf-8')]

devLabel = json.load(open('../../data/Dev_targets.json', 'r'))

TP = 0.0
TN = 0.0
FP = 0.0
P = 0.0
R = 0.0
F1 = 0.0
for code in devLabel:
    tmp = [x for x, p in devLabel[code]]
    temp = set(tmp)
    if code not in label:
        TN += len(temp)
    else:
        for target in label[code]:
            if target not in temp:
                FP += 1
            else:
                TP += 1
        for target in temp:
            if target not in label[code]:
                TN += 1
P = TP/(TP+TN)
R = TP/(TP+FP)
F1 = 2*P*R/(P+R)
print 'P:\t%f'%P
print 'R:\t%f'%R
print 'F1:\t%f'%F1
