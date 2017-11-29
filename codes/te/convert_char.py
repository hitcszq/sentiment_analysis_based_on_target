#! -*- coding:utf8 -*-
#将原始文本(未分词)处理成需要的格式
import sys, json

#产生char2id.txt
def genChar2Id():
    id2Char = []
    with open('../../data/Raw.csv', 'r') as f:
        f.next()
        tmp = set([])
        for line in f:
            temp = line.strip().split('\t')
            temp[1] = temp[1].decode('utf-8')
            for char in temp[1]:
                tmp.add(char)
        id2Char += list(tmp)

    with open('../../data/Test.csv', 'r') as f:
        f.next()
        tmp = set([])
        for line in f:
            temp = line.strip().split('\t')
            temp[1] = temp[1].decode('utf-8')
            for char in temp[1]:
                tmp.add(char)
        id2Char += list(tmp)

    id2Char = list(set(id2Char))

    with open('../../data/char2id.txt', 'w') as f:
        flag = 0
        for index, char in enumerate(id2Char):
            if flag == 0:
                f.write(char.encode('utf-8')+'\t'+str(index))
                flag = 1
            else:
                f.write('\n'+char.encode('utf-8')+'\t'+str(index))

char2Id = {}
def readChar2Id():
    with open('../../data/char2id.txt', 'r') as f:
        for line in f:
            temp = line.rstrip().split('\t')
            char2Id[temp[0]] = int(temp[1])


data = {}
dataPreWs = {}
dataPrePos = {}
label = {}

def train_convert(codeFile, tagFile):
    global data, label, char2Id
    flag = 0
    for code in data:
        if code not in label:
            if flag == 0:
                codeFile.write(code)
                flag = 1
            else:
                codeFile.write('\n'+code)
            tagSentence = ['O' for i in range(len(sentence))]
            for char, tag in zip(sentence, tagSentence):
                tagFile.write('%s %s\n'%(str(char2Id[char.encode('utf-8')]), tag))
            tagFile.write('\n')
            continue
        tp = []
        sentence = data[code]
        i = 0
        while i < len(sentence):
            k = 50
            while k >= 0:
                temp = ''
                j = i
                while j < len(sentence) and j <= i+k:
                    temp += sentence[j].encode('utf-8')
                    j += 1
                if temp in label[code]:
                    if i+k <= len(sentence)-1:
                        tp += [(temp, (i, i+k))]
                    else:
                        tp += [(temp, (i, len(sentence)-1))]
                    i = j-1
                    break
                k -= 1
            i += 1
        if tp:
            if flag == 0:
                codeFile.write(code)
                flag = 1
            else:
                codeFile.write('\n'+code)
            tagSentence = ['O' for i in range(len(sentence))]
            for t, p in tp:
                tagSentence[p[0]] = 'B'
                tagSentence[p[0]+1:p[1]+1] = ['I' for i in range(p[1]-p[0])]
            for char, tag in zip(sentence, tagSentence):
                tagFile.write('%s %s\n'%(str(char2Id[char.encode('utf-8')]), tag))
            tagFile.write('\n')

OpenNer = set([])

def train_convert_1(codeFile, tagFile):
    global data, dataPreWs, dataPrePos, OpenNer, char2Id
    flag = 0
    for code in data:
        tp = []
        tp_1 = []
        sentence = data[code]

        if code not in label:
            i = 0
            while i < len(sentence):
                k = 50
                while k >= 0:
                    temp = ''
                    j = i
                    while j < len(sentence) and j <= i+k:
                        temp += sentence[j].encode('utf-8')
                        j += 1
                    if temp in OpenNer:
                        if i+k <= len(sentence) - 1:
                            tp_1 += [(temp, (i, i+k))]
                        else:
                            tp_1 += [(temp, (i, len(sentence)-1))]
                        i = j-1
                        break
                    k -= 1
                i += 1
            if flag == 0:
                codeFile.write(code)
                flag = 1
            else:
                codeFile.write('\n'+code)
            tagSentence = ['O' for i in range(len(sentence))]
            tagSentence_1 = ['O' for i in range(len(sentence))]

            for t, p in tp_1:
                tagSentence_1[p[0]] = 'B'
                #tagSentence_1[p[0]+1:p[1]+1] = ['I' for i in range(p[1]-p[0])]
                if p[1] - p[0] + 1 > 2:
                    tagSentence_1[p[0]+1:p[1]] = ['I' for i in range(p[1]-p[0]-1)]
                if p[1] - p[0] > 0:
                    tagSentence_1[p[1]] = 'E'
            for char, tag, tag_1, ws, pos in zip(sentence, tagSentence, tagSentence_1, dataPreWs[code], dataPrePos[code]):
                tagFile.write('%s %s %s %s %s\n'%(str(char2Id[char.encode('utf-8')]), ws, pos, tag_1, tag))
            tagFile.write('\n')
            continue

        i = 0
        while i < len(sentence):
            k = 50
            while k >= 0:
                temp = ''
                j = i
                while j < len(sentence) and j <= i+k:
                    temp += sentence[j].encode('utf-8')
                    j += 1
                if temp in label[code]:
                    if i+k <= len(sentence)-1:
                        tp += [(temp, (i, i+k))]
                    else:
                        tp += [(temp, (i, len(sentence)-1))]
                    i = j-1
                    break
                k -= 1
            i += 1

        i = 0
        while i < len(sentence):
            k = 50
            while k >= 0:
                temp = ''
                j = i
                while j < len(sentence) and j <= i+k:
                    temp += sentence[j].encode('utf-8')
                    j += 1
                if temp in OpenNer:
                    if i+k <= len(sentence) - 1:
                        tp_1 += [(temp, (i, i+k))]
                    else:
                        tp_1 += [(temp, (i, len(sentence)-1))]
                    i = j-1
                    break
                k -= 1
            i += 1

        if tp:
            if flag == 0:
                codeFile.write(code)
                flag = 1
            else:
                codeFile.write('\n'+code)
            tagSentence = ['O' for i in range(len(sentence))]
            tagSentence_1 = ['O' for i in range(len(sentence))]
            for t, p in tp:
                tagSentence[p[0]] = 'B'
                #tagSentence[p[0]+1:p[1]+1] = ['I' for i in range(p[1]-p[0])]
                if p[1] - p[0] + 1 > 2:
                    tagSentence[p[0]+1:p[1]] = ['I' for i in range(p[1]-p[0]-1)]
                if p[1] - p[0] > 0:
                    tagSentence[p[1]] = 'E'

            for t, p in tp_1:
                tagSentence_1[p[0]] = 'B'
                if p[1] - p[0] + 1 > 2:
                    tagSentence_1[p[0]+1:p[1]] = ['I' for i in range(p[1]-p[0]-1)]
                if p[1] - p[0] > 0:
                    tagSentence_1[p[1]] = 'E'

                #tagSentence_1[p[0]+1:p[1]+1] = ['I' for i in range(p[1]-p[0])]
            for char, tag, tag_1, ws, pos in zip(sentence, tagSentence, tagSentence_1, dataPreWs[code], dataPrePos[code]):
                tagFile.write('%s %s %s %s %s\n'%(str(char2Id[char.encode('utf-8')]), ws, pos, tag_1, tag))
            tagFile.write('\n')

testData = {}
def test_convert_1(codeFile, tagFile):
    global testData, OpenNer, char2Id
    flag = 0
    for code in testData:
        tp_1 = []
        sentence = testData[code]

        i = 0
        while i < len(sentence):
            k = 50
            while k >= 0:
                temp = ''
                j = i
                while j < len(sentence) and j <= i+k:
                    temp += sentence[j].encode('utf-8')
                    j += 1
                if temp in OpenNer:
                    if i+k <= len(sentence) - 1:
                        tp_1 += [(temp, (i, i+k))]
                    else:
                        tp_1 += [(temp, (i, len(sentence)-1))]
                    i = j-1
                    break
                k -= 1
            i += 1

        if flag == 0:
            codeFile.write(code)
            flag = 1
        else:
            codeFile.write('\n'+code)
        tagSentence = ['O' for i in range(len(sentence))]
        tagSentence_1 = ['O' for i in range(len(sentence))]
        for t, p in tp_1:
            tagSentence_1[p[0]] = 'B'
            #tagSentence_1[p[0]+1:p[1]+1] = ['I' for i in range(p[1]-p[0])]

            if p[1] - p[0] + 1 > 2:
                tagSentence_1[p[0]+1:p[1]] = ['I' for i in range(p[1]-p[0]-1)]
            if p[1] - p[0] > 0:
                tagSentence_1[p[1]] = 'E'

        for char, tag, tag_1, ws, pos in zip(sentence, tagSentence, tagSentence_1, dataPreWs[code], dataPrePos[code]):
            tagFile.write('%s %s %s %s %s\n'%(str(char2Id[char.encode('utf-8')]), ws, pos, tag_1, tag))
        tagFile.write('\n')
if __name__ == '__main__':
    #产生char2id.txt
    #genChar2Id()

    #读入char2id.txt
    readChar2Id()

    #读入已有的品牌
    with open('../../data/new_View.csv', 'r') as f:
        f.next()
        for line in f:
            temp = line.rstrip().split('\t')
            OpenNer.add(temp[1])
    with open('../../data/Label.csv', 'r') as f:
        f.next()
        for line in f:
            temp = line.strip().split('\t')
            OpenNer.add(temp[1])
    with open('../prepare/extra_car_dict.txt', 'r') as f:
        for line in f:
            temp = line.strip()
            OpenNer.add(temp)
    
    dirname = '../../data/'
    filename = sys.argv[1]
    if filename == 'Train' or filename == 'Dev' or filename == 'Raw':
        '''
        转换训练数据和验证数据
        '''
        dataFile = open(dirname+filename+'.csv', 'r')
        dataFile.next()
        for line in dataFile:
            temp = line.strip().split('\t')
            temp[1] = temp[1].decode('utf-8')
            data[temp[0]] = temp[1]
        dataFile.close()

        dataPreWs = json.load(open(dirname+filename+'_pre_ws.json', 'r'))
        dataPrePos = json.load(open(dirname+filename+'_pre_pos.json', 'r'))
        
        labelFile = open(dirname+'Label.csv', 'r')
        labelFile.next()
        for line in labelFile:
            temp = line.strip().split('\t')
            if temp[0] not in label:
                label[temp[0]] = [temp[1]]
            else:
                label[temp[0]] += [temp[1]]
        labelFile.close()

        codeFile = open(dirname+filename+'_char_code.txt', 'w')
        tagFile = open(dirname+filename+'_char_tag.txt', 'w')

        train_convert_1(codeFile, tagFile)

        tagFile.close()
        codeFile.close()
    elif filename == 'Test':
        '''
        转换测试数据
        '''
        dataFile = open(dirname+filename+'.csv', 'r')
        dataFile.next()
        for line in dataFile:
            temp = line.strip().split('\t')
            temp[1] = temp[1].decode('utf-8')
            testData[temp[0]] = temp[1]
        dataFile.close()
        dataPreWs = json.load(open(dirname+filename+'_pre_ws.json', 'r'))
        dataPrePos = json.load(open(dirname+filename+'_pre_pos.json', 'r'))
  
        codeFile = open(dirname+filename+'_char_code.txt', 'w')
        tagFile = open(dirname+filename+'_char_tag.txt', 'w')
    
        test_convert_1(codeFile, tagFile)
    
        tagFile.close()
        codeFile.close()
