# coding:utf-8
'''
将Test_char_code.txt和Test_char_tag_pred.txt合并得到抽取的结果
'''
import json, sys

dirname = '../../data/'
filename = sys.argv[1]

with open(dirname+filename+'.csv', 'r') as f:
    rawData = {}
    f.next()
    for line in f:
        temp = line.strip().split('\t')
        temp[1] = temp[1].decode('utf-8')
        rawData[temp[0]] = temp[1]


#读取Test_tag_pred.txt的迭代器
def readTTP(ttpFile):
    ret = []
    for line in ttpFile:
        line = line.strip()
        if not line:
            yield ret
            ret = []
        else:
            ret += [line]

tagData = {}
codeFile = open(dirname+filename+'_char_code.txt', 'r')
ttpFile = open(dirname+filename+'_char_tag_pred.txt', 'r')

code = codeFile.next()

ttpIter = readTTP(ttpFile)
ttp = ttpIter.next()

flag = 0
cnt = 0
code_targets = {}
while code and ttp:
    code = code.strip()
    i = 0
    targets = []
    target = ''
    pre = 0
    while i < len(ttp):
        if ttp[i] == 'B':
            if target:
                targets += [(target, (pre, i-1))]
            target = ''
            pre = i
            target += rawData[code][i].encode('utf-8')
        elif ttp[i] == 'I' or ttp[i] == 'E':
            target += rawData[code][i].encode('utf-8')
        else:
            if target:
                targets += [(target, (pre, i-1))]
            target = ''
        i += 1
    if target:
        targets += [(target, (pre, i-1))]
    code_targets[code] = targets
    #if flag == 0:
    #    ttFile.write(code)
    #    flag = 1
    #else:
    #    ttFile.write('\n'+code)
    #for t in targets:
    #    cnt += 1
    #    ttFile.write('\t'+t.encode('utf-8'))
    try:
        code = codeFile.next()
        ttp = ttpIter.next()
    except:
        break
json.dump(code_targets, open(dirname+filename+'_char_targets.json', 'w'))
ttpFile.close()
codeFile.close()
