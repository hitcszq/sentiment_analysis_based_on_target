# coding:utf-8
import jieba, json, random
import jieba.posseg as pseg
import sys

#将训练数据分成train和dev两部分
def partition(rawFile, trainFile, devFile):
    data = []
    for line in rawFile:
        temp = line.strip().split('\t')
        data += [(temp[0], temp[1])]
    random.shuffle(data)

    trainFile.write('SentenceId\tContent')
    devFile.write('SentenceId\tContent')

    cnt = 0
    for code, cont in data:
        cnt += 1
        if cnt <=3500:
            devFile.write('\n'+code+'\t'+cont)
        else:
            trainFile.write('\n'+code+'\t'+cont)

def splitLabel(labelFile, trainLabelFile, devLabelFile, trainFile, devFile):
    trainCode = []
    trainFile.next()
    for line in trainFile:
        temp = line.strip().split('\t')
        trainCode += [temp[0]]

    devCode = []
    devFile.next()
    for line in devFile:
        temp = line.strip().split('\t')
        devCode += [temp[0]]

    label = {}
    labelFile.next()
    for line in labelFile:
        temp = line.strip().split('\t')
        if temp[0] not in label:
            label[temp[0]] = [(temp[1], temp[2])]
        else:
            label[temp[0]] += [(temp[1], temp[2])]

    trainLabelFile.write('SentenceId\tView\tOpinion')
    devLabelFile.write('SentenceId\tView\tOpinion')

    for code in trainCode:
        if code in label:
            for view, opinion in label[code]:
                trainLabelFile.write('\n'+code+'\t'+view+'\t'+opinion)

    for code in devCode:
        if code in label:
            for view, opinion in label[code]:
                devLabelFile.write('\n'+code+'\t'+view+'\t'+opinion)

#分词
def ws(inputFile):
    jieba.load_userdict('extra_car_dict.txt')
    res = {}
    inputFile.next()
    cnt = 0
    for line in inputFile:
        cnt += 1
        print('正在处理第%d条'%cnt)
        temp = line.strip().split('\t')
        if len(temp) != 2:
            continue
        res[temp[0]] = jieba.lcut(temp[1])
        print('第%d条处理完毕'%cnt)
    return res

#分词+词性标注
def pos(inputFile):
    #jieba.load_userdict('extra_car_dict.txt')
    res_ws = {}
    res_pos = {}
    inputFile.next()
    for line in inputFile:
        temp = line.strip().split('\t')
        if len(temp) != 2:
            print 'error', temp[0], len(temp)
            continue
        words = pseg.cut(temp[1])
        res_ws[temp[0]] = []
        res_pos[temp[0]] = []
        for word, pos in words:
            if len(word) == 1:
                res_ws[temp[0]] += ['S']
            elif len(word) == 2:
                res_ws[temp[0]] += ['B', 'E']
            else:
                res_ws[temp[0]] += ['B']
                res_ws[temp[0]] += ['I']*(len(word)-2)
                res_ws[temp[0]] += ['E']
            res_pos[temp[0]] += [pos]*len(word)
    return res_ws, res_pos

if __name__ == '__main__':
    
    '''
    将原始训练集切分成训练和验证两部分
    '''
    '''
    rawFile = open('../../data/Raw.csv', 'r')
    trainFile = open('../../data/Train.csv', 'w')
    devFile = open('../../data/Dev.csv', 'w')
    rawFile.next()

    partition(rawFile, trainFile, devFile)

    devFile.close()
    trainFile.close()
    rawFile.close()
    '''
    '''
    将原始Label分成训练和验证两部分
    '''
    '''
    labelFile = open('../../data/Label.csv', 'r')
    trainLabelFile = open('../../data/Label_Gold_Train.csv', 'w')
    devLabelFile = open('../../data/Label_Gold_Dev.csv', 'w')
    trainFile = open('../../data/Train.csv', 'r')
    devFile = open('../../data/Dev.csv', 'r')
    
    splitLabel(labelFile, trainLabelFile, devLabelFile, trainFile, devFile)

    devFile.close()
    trainFile.close()
    devLabelFile.close()
    trainLabelFile.close()
    labelFile.close()
    '''
    '''
    分词
    '''
    '''
    #rawFile = open('../data/Raw.csv', 'r')
    trainFile = open('../data/Train.csv', 'r')
    devFile = open('../data/Dev.csv', 'r')

    #raw = ws(rawFile)
    train = ws(trainFile)
    dev = ws(devFile)

    devFile.close()
    trainFile.close()
    #rawFile.close()

    #rawFile1 = open('../data/Raw_ws.json', 'w')
    trainFile1 = open('../data/Train_ws.json', 'w')
    devFile1 = open('../data/Dev_ws.json', 'w')

    #json.dump(raw, rawFile1)
    json.dump(train, trainFile1)
    json.dump(dev, devFile1)

    devFile1.close()
    trainFile1.close()
    #rawFile1.close()

    testFile = open('../data/Test.csv', 'r')
    test = ws(testFile)
    testFile.close()

    testFile1 = open('../data/Test_ws.json', 'w')
    json.dump(test, testFile1)
    testFile1.close()
    '''
    
    '''
    分词+词性标注
    '''
    filename = sys.argv[1]
    if filename == 'Raw':
        rawFile = open('../../data/Raw.csv', 'r')
    #trainFile = open('../../data/Train.csv', 'r')
    #devFile = open('../../data/Dev.csv', 'r')

        raw_ws, raw_pos = pos(rawFile)
    #train_ws, train_pos = pos(trainFile)
    #dev_ws, dev_pos = pos(devFile)

    #devFile.close()
    #trainFile.close()
        rawFile.close()

        rawFile1 = open('../../data/Raw_pre_ws.json', 'w')
        rawFile2 = open('../../data/Raw_pre_pos.json', 'w')
    #trainFile1 = open('../../data/Train_pre_ws.json', 'w')
    #devFile1 = open('../../data/Dev_pre_ws.json', 'w')
    #trainFile2 = open('../../data/Train_pre_pos.json', 'w')
    #devFile2 = open('../../data/Dev_pre_pos.json', 'w')


        json.dump(raw_ws, rawFile1)
        json.dump(raw_pos, rawFile2)
    #json.dump(train_ws, trainFile1)
    #json.dump(dev_ws, devFile1)
    #json.dump(train_pos, trainFile2)
    #json.dump(dev_pos, devFile2)

    #devFile1.close()
    #trainFile1.close()
    #devFile2.close()
    #trainFile2.close()
        rawFile1.close()
        rawFile2.close()
    elif filename == 'Test':
        testFile = open('../../data/Test.csv', 'r')
        test_ws, test_pos = pos(testFile)
        testFile.close()

        testFile1 = open('../../data/Test_pre_ws.json', 'w')
        testFile2 = open('../../data/Test_pre_pos.json', 'w')
        json.dump(test_ws, testFile1)
        json.dump(test_pos, testFile2)
        testFile1.close()
        testFile2.close()
