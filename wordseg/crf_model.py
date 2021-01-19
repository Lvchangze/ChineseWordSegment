import torch  # 仅用torch.save来存model，没有使用其他函数


count = 0


class CRFModel:
    def __init__(self):
        self.scoreMap = {}
        self.UnigramTemplates = []
        self.BigramTemplates = []
        self.readTemplate()

    def readTemplate(self):
        tempFile = open("dataset/dataset2/template.utf8", encoding='utf-8')
        switchFlag = False  # 先读Unigram，在读Bigram
        for line in tempFile:
            tmpList = []
            if line.find("Unigram") > 0 or line.find("Bigram") > 0:  # 读到'Unigram'或者'Bigram'
                continue
            if switchFlag:
                if line.find("/") > 0:
                    left = line.split("/")[0].split("[")[-1].split(",")[0]
                    right = line.split("/")[-1].split("[")[-1].split(",")[0]
                    tmpList.append(int(left))
                    tmpList.append(int(right))
                else:
                    num = line.split("[")[-1].split(",")[0]
                    tmpList.append(int(num))
                self.BigramTemplates.append(tmpList)
            else:
                if len(line.strip()) == 0:
                    switchFlag = True
                else:
                    if line.find("/") > 0:
                        left = line.split("/")[0].split("[")[-1].split(",")[0]
                        right = line.split("/")[-1].split("[")[-1].split(",")[0]
                        tmpList.append(int(left))
                        tmpList.append(int(right))
                    else:
                        num = line.split("[")[-1].split(",")[0]
                        tmpList.append(int(num))
                    self.UnigramTemplates.append(tmpList)

    def getTrainData(self):
        sentences = []
        results = []
        tempFile = open('dataset/dataset2/train.utf8', encoding='utf-8')
        sentence = ""
        result = ""
        for line in tempFile:
            line = line.strip()
            if line == "":
                if sentence == "" or result == "":
                    pass
                else:
                    sentences.append(sentence)
                    results.append(result)
                sentence = ""
                result = ""
            else:
                data = line.split(" ")
                sentence += data[0]
                result += data[1]
        return [sentences, results]

    # 找出一句句子中，给定的模板下的某位置的标注（BIES）
    def makeKey(self, template, identity, sentence, pos, statusCovered):
        result = ""
        result += identity
        for i in template:
            index = pos + i
            if index < 0 or index >= len(sentence):
                result += " "
            else:
                result += sentence[index]
        result += "/"
        result += statusCovered
        # print(result)
        return result

    # 用于计算正确率
    def getWrongNum(self, sentence, realRes):
        myRes = self.Viterbi(sentence)  # 我的解
        lens = len(sentence)
        wrongNum = 0
        for i in range(0, lens):
            myResI = myRes[i]  # 我的解
            theoryResI = realRes[i]  # 理论解
            if myResI != theoryResI:
                wrongNum += 1
        return wrongNum

    def calScoreMap(self, sentence, realRes):
        myRes = self.Viterbi(sentence)  # 我的解
        for i in range(0, len(sentence)):
            myResI = myRes[i]  # 我的解
            theoryResI = realRes[i]  # 理论解
            if myResI != theoryResI:  # 如果和理论值不同

                # print("Unigram更新开始")
                uniTem = self.UnigramTemplates
                for uniIndex in range(0, len(uniTem)):
                    print(uniTem[uniIndex])
                    print(str(uniIndex))
                    print(sentence)
                    print(myResI)
                    uniMyKey = self.makeKey(uniTem[uniIndex], str(uniIndex), sentence, i, myResI)  # 我的标注
                    if uniMyKey not in self.scoreMap:
                        self.scoreMap[uniMyKey] = -1
                    else:
                        self.scoreMap[uniMyKey] = self.scoreMap[uniMyKey] - 1
                    uniTheoryKey = self.makeKey(uniTem[uniIndex], str(uniIndex), sentence, i, theoryResI)  # 正确的标注
                    if uniTheoryKey not in self.scoreMap:
                        self.scoreMap[uniTheoryKey] = 1
                    else:
                        self.scoreMap[uniTheoryKey] = self.scoreMap[uniTheoryKey] + 1

                # print("Bigram更新开始")
                biTem = self.BigramTemplates
                for biIndex in range(0, len(biTem)):
                    if i == 0:
                        biMyKey = self.makeKey(biTem[biIndex], str(biIndex), sentence, i, " " + str(myResI))
                        biTheoryKey = self.makeKey(biTem[biIndex], str(biIndex), sentence, i, " " + str(theoryResI))
                    else:
                        biMyKey = self.makeKey(biTem[biIndex], str(biIndex), sentence, i, myRes[i - 1:i + 1:])
                        biTheoryKey = self.makeKey(biTem[biIndex], str(biIndex), sentence, i, myRes[i - 1:i + 1:])
                    if biMyKey not in self.scoreMap:
                        self.scoreMap[biMyKey] = -1
                    else:
                        self.scoreMap[biMyKey] = self.scoreMap[biMyKey] - 1
                    if biTheoryKey not in self.scoreMap:
                        self.scoreMap[biTheoryKey] = 1
                    else:
                        self.scoreMap[biTheoryKey] = self.scoreMap[biTheoryKey] + 1

    def getUnigramScore(self, sentence, thisPos, thisStatus):
        unigramScore = 0
        unigramTemplates = self.UnigramTemplates
        for i in range(0, len(unigramTemplates)):
            key = self.makeKey(unigramTemplates[i], str(i), sentence, thisPos, thisStatus)
            if key in self.scoreMap:
                unigramScore += self.scoreMap[key]
        return unigramScore

    def getBigramScore(self, sentence, thisPos, preStatus, thisStatus):
        bigramScore = 0
        bigramTemplates = self.BigramTemplates
        for i in range(0, len(bigramTemplates)):
            key = self.makeKey(bigramTemplates[i], str(i), sentence, thisPos, preStatus + thisStatus)
            if key in self.scoreMap:
                bigramScore += self.scoreMap[key]
        return bigramScore

    def num2Tag(self, row):
        if row == 0:
            return "B"
        elif row == 1:
            return "I"
        elif row == 2:
            return "E"
        elif row == 3:
            return "S"
        else:
            return None

    def tag2Num(self, status):
        if status == "B":
            return 0
        elif status == "I":
            return 1
        elif status == "E":
            return 2
        elif status == "S":
            return 3
        else:
            return -1

    def getMaxIndex(self, list):
        origin = list.copy()
        origin.sort()
        max = origin[-1]
        index = list.index(max)
        return index

    # 状态序列里，正确的状态的个数
    def getDuplicate(self, s1, s2):
        length = min(len(s1), len(s2))
        count = 0
        for i in range(0, length):
            if s1[i] == s2[i]:
                count += 1
        return count

    def Viterbi(self, sentence):
        lens = len(sentence)
        statusFrom = [[""] * lens, [""] * lens, [""] * lens, [""] * lens]  # B,I,E,S
        maxScore = [[0] * lens, [0] * lens, [0] * lens, [0] * lens]  # 4条路
        for word in range(0, lens):
            for stateNum in range(0, 4):
                thisStatus = self.num2Tag(stateNum)
                # 第一个词
                if word == 0:
                    uniScore = self.getUnigramScore(sentence, 0, thisStatus)
                    biScore = self.getBigramScore(sentence, 0, " ", thisStatus)
                    maxScore[stateNum][0] = uniScore + biScore
                    statusFrom[stateNum][0] = None
                else:
                    scores = [0] * 4
                    for i in range(0, 4):
                        preStatus = self.num2Tag(i)
                        transScore = maxScore[i][word - 1]
                        uniScore = self.getUnigramScore(sentence, word, thisStatus)
                        biScore = self.getBigramScore(sentence, word, preStatus, thisStatus)
                        scores[i] = transScore + uniScore + biScore
                    maxIndex = self.getMaxIndex(scores)
                    maxScore[stateNum][word] = scores[maxIndex]
                    statusFrom[stateNum][word] = self.num2Tag(maxIndex)
        resBuf = [""] * lens
        scoreBuf = [0] * 4
        if lens > 0:
            for i in range(0, 4):
                scoreBuf[i] = maxScore[i][lens - 1]
            resBuf[lens - 1] = self.num2Tag(self.getMaxIndex(scoreBuf))
        for backIndex in range(lens - 2, -1, -1):
            resBuf[backIndex] = statusFrom[self.tag2Num(resBuf[backIndex + 1])][backIndex + 1]
        res = "".join(resBuf)
        return res

    def myTrain(self):
        sentences, results = self.getTrainData()
        whole = len(sentences)
        trainNum = int(whole * 0.8)
        for epoch in range(1, 15):
            wrongNum = 0
            totalTest = 0
            for i in range(0, trainNum):
                sentence = sentences[i]
                totalTest += len(sentence)
                result = results[i]
                self.calScoreMap(sentence, result)  # 训练的关键，计算scoreMap
                wrongNum += self.getWrongNum(sentence, result)
            correctNum = totalTest - wrongNum
            print("epoch" + str(epoch) + ":准确率" + str(float(correctNum / totalTest)))
            total = 0
            correct = 0
            # 测试集为后20%
            for i in range(trainNum, whole):
                sentence = sentences[i]
                total += len(sentence)
                result = results[i]
                myRes = self.Viterbi(sentence)
                correct += self.getDuplicate(result, myRes)
            accuracy = float(correct / total)
            print("accuracy" + str(accuracy))
            torch.save(
                {
                    'scoreMap': self.scoreMap,
                    'BigramTemplates': self.BigramTemplates,
                    'UnigramTemplates': self.UnigramTemplates
                },
                "../models/CRF-dataSet.model"
            )

    def predict(self, sentence, checkpoint):
        global count
        count += 1
        # print(count)
        self.scoreMap = checkpoint['scoreMap']
        self.UnigramTemplates = checkpoint['UnigramTemplates']
        self.BigramTemplates = checkpoint['BigramTemplates']
        return self.Viterbi(sentence)


if __name__ == '__main__':
    model = CRFModel()
    model.myTrain()
    # print(model.predict("我觉得你很强，是永远的神"))
