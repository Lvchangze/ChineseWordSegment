import numpy as np
import torch  # 仅用torch.save来存model，没有使用其他函数


class HMMModel:
    def __init__(self):
        self.stateSet = ['B', 'I', 'E', 'S']
        self.stateTransitionMatrix = {}  # 状态转移概率矩阵
        self.emissionMatrix = {}  # 发射概率矩阵
        self.initStateMatrix = {}  # 初始状态矩阵
        self.stateCount = {}  # ‘B,I,E,S’每个状态在训练集中出现的次数
        self.trainSentenceNum = 0  # 训练集语句数量

    # 初始化所有概率矩阵
    def initArray(self):
        # 初始化状态转移矩阵
        for state0 in self.stateSet:
            self.stateTransitionMatrix[state0] = {}
            for state1 in self.stateSet:
                self.stateTransitionMatrix[state0][state1] = 0.0
        # 初始化状态矩阵，初始化发射矩阵，初始化状态计数
        for state in self.stateSet:
            self.initStateMatrix[state] = 0.0  # initStateMatrix['B'] = 0.0等
            self.emissionMatrix[state] = {}
            self.stateCount[state] = 0

    def getTrainSet(self):
        trainSet = {}  # 字典，句子对应状态
        tmpSentence = ''
        tmpSentenceState = ''
        preNull = False
        with open('../dataset/dataset2/train.utf8', encoding='utf-8') as trainFile:
            while True:
                s = trainFile.readline()
                if s == "":  # 文件读完
                    break
                s = s.strip()  # 去掉头尾空格
                if s == "":  # 读到换行符
                    if not preNull:
                        trainSet[tmpSentence] = tmpSentenceState
                        tmpSentence = ''
                        tmpSentenceState = ''
                    preNull = True
                    continue
                preNull = False
                s = s.replace(" ", "")
                tmpSentence += s[0]
                tmpSentenceState += s[1]
        with open('../dataset/dataset1/train.utf8', encoding='utf-8') as trainFile:
            while True:
                s = trainFile.readline()
                if s == "":  # 文件读完
                    break
                s = s.strip()  # 去掉头尾空格
                if s == "":  # 读到换行符
                    if not preNull:
                        trainSet[tmpSentence] = tmpSentenceState
                        tmpSentence = ''
                        tmpSentenceState = ''
                    preNull = True
                    continue
                preNull = False
                s = s.replace(" ", "")
                tmpSentence += s[0]
                tmpSentenceState += s[1]
        print(len(trainSet))
        print(trainSet)
        return trainSet

    # 将参数估计的概率取对数，则概率1取0，,概率0取无穷小，为-3.14e+100
    def arrayToProb(self):
        # 初始状态概率
        for key in self.initStateMatrix:
            if self.initStateMatrix[key] == 0.0:
                self.initStateMatrix[key] = -3.14e+100
            else:
                self.initStateMatrix[key] = np.log(self.initStateMatrix[key] / self.trainSentenceNum)
        # 状态转移概率
        for key0 in self.stateTransitionMatrix:
            for key1 in self.stateTransitionMatrix[key0]:
                if self.stateTransitionMatrix[key0][key1] == 0.0:
                    self.stateTransitionMatrix[key0][key1] = -3.14e+100
                else:
                    self.stateTransitionMatrix[key0][key1] = np.log(
                        self.stateTransitionMatrix[key0][key1] / self.stateCount[key0])
        # 发射概率
        for key in self.emissionMatrix:
            for word in self.emissionMatrix[key]:
                if self.emissionMatrix[key][word] == 0.0:
                    self.emissionMatrix[key][word] = -3.14e+100
                else:
                    self.emissionMatrix[key][word] = np.log(self.emissionMatrix[key][word] / self.stateCount[key])

    def train(self):
        self.initArray()
        trainSet = self.getTrainSet()  # 字典，句子:状态
        # print(trainSet)
        for sent in trainSet:
            self.trainSentenceNum += 1
            wordList = []  # list类型，一个字一个entry
            for i in range(len(sent)):
                wordList.extend(sent[i])
            lineState = []  # 句子的状态序列，list类型
            for i in range(len(trainSet[sent])):
                lineState.extend(trainSet[sent][i])
            # 统计初始状态分布概率
            self.initStateMatrix[lineState[0]] += 1
            # 统计状态转移概率
            for j in range(len(lineState) - 1):
                self.stateTransitionMatrix[lineState[j]][lineState[j + 1]] += 1
            # 统计状态计数和发射概率
            for p in range(len(lineState)):
                self.stateCount[lineState[p]] += 1  # 记录每一个状态的出现次数
                for state in self.stateSet:
                    if wordList[p] not in self.emissionMatrix[state]:  # 将wordList[p]这个字加入发射概率矩阵
                        self.emissionMatrix[state][wordList[p]] = 0.0
                # 计算发射概率矩阵
                self.emissionMatrix[lineState[p]][wordList[p]] += 1
        self.arrayToProb()
        torch.save(self, '../models/hmm.model')

    # Viterbi算法求测试集最优状态序列
    def Viterbi(self, sentence, initStateMatrix, stateTransitionMatrix, emissionMatrix):
        tab = [{}]  # 动态规划表
        path = {}  # 存路径

        # 若第一个字没有出现在发射矩阵的'B'状态列表上，则默认他为S，所以其他状态的概率都为负无穷大
        if sentence[0] not in emissionMatrix['B']:
            for state in self.stateSet:
                if state == 'S':
                    emissionMatrix[state][sentence[0]] = 0
                else:
                    emissionMatrix[state][sentence[0]] = -3.14e+100

        # path总共4条路
        for state in self.stateSet:
            # tab[t][state]表示时刻t到达state状态的所有路径中，概率最大路径的概率值
            # 计算的时候是相加而不是相乘，因为之前取过对数的原因
            tab[0][state] = initStateMatrix[state] + emissionMatrix[state][sentence[0]]
            path[state] = [state]

        # 对句子中的每个字进行判断
        for i in range(1, len(sentence)):
            tab.append({})
            new_path = {}
            # 新增begin标识和end标识，一个词汇的开始与结束
            for state in self.stateSet:
                if state == 'B':
                    emissionMatrix[state]['begin'] = 0
                else:
                    emissionMatrix[state]['begin'] = -3.14e+100
            for state in self.stateSet:
                if state == 'E':
                    emissionMatrix[state]['end'] = 0
                else:
                    emissionMatrix[state]['end'] = -3.14e+100
            # 开始计算
            for state0 in self.stateSet:
                items = []
                for state1 in self.stateSet:
                    # 若这个字在当前状态的发射矩阵中不存在
                    if sentence[i] not in emissionMatrix[state0]:
                        # 前一个字也不在当前状态的发射矩阵中
                        if sentence[i - 1] not in emissionMatrix[state0]:
                            prob = tab[i - 1][state1] \
                                   + stateTransitionMatrix[state1][state0] \
                                   + emissionMatrix[state0]['end']
                        # 前一个字在当前状态的发射矩阵中
                        else:
                            prob = tab[i - 1][state1] \
                                   + stateTransitionMatrix[state1][state0] \
                                   + emissionMatrix[state0]['begin']
                    # 这个字在发射矩阵中，就计算每个字符对应STATES的概率
                    else:
                        prob = tab[i - 1][state1] \
                               + stateTransitionMatrix[state1][state0] \
                               + emissionMatrix[state0][sentence[i]]
                    items.append((prob, state1))
                best = max(items)
                tab[i][state0] = best[0]
                new_path[state0] = path[best[1]] + [state0]
            path = new_path
        # print(path)
        # print([(tab[len(sentence) - 1][state], state) for state in self.stateSet])
        prob, state = max([(tab[len(sentence) - 1][state], state) for state in self.stateSet])
        str = ''
        for i in path[state]:
            str += i
        return str


def predict(sentence, model):
    return model.Viterbi(sentence, model.initStateMatrix, model.stateTransitionMatrix, model.emissionMatrix)


if __name__ == '__main__':
    model = HMMModel()
    # model.train()
    # print(predict("我是梁志超他奶奶"))
