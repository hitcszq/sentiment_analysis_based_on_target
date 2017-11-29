#coding:utf-8
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
if __name__ == '__main__':
    #产生char2id.txt
    genChar2Id()
