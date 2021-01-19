import torch
from torch import nn
from torch import optim


class BiLSTM_CRF(nn.Module):
    def __init__(self):
        super(BiLSTM_CRF, self).__init__()
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        self.tag_to_ix = {"B": 0, "I": 1, "E": 2, "S": 3, self.START_TAG: 4, self.STOP_TAG: 5}
        self.embedding_dim = 300
        self.hidden_dim = 150
        self.epochs = 100

        self.trainSet = {}  # 句子对应状态
        tmpSentence = ''
        tmpSentenceState = ''
        preNull = False
        with open('dataset/dataset1/train.utf8', encoding='utf-8') as trainFile:
            while True:
                s = trainFile.readline()
                if s == "":  # 文件读完
                    break
                s = s.strip()  # 去掉头尾空格
                if s == "":  # 读到换行符
                    if not preNull:
                        self.trainSet[tmpSentence] = tmpSentenceState
                        tmpSentence = ''
                        tmpSentenceState = ''
                    preNull = True
                    continue
                preNull = False
                s = s.replace(" ", "")
                tmpSentence += s[0]
                tmpSentenceState += s[1]
        # print(self.trainSet)
        content = []
        label = []
        for key in self.trainSet:
            tmpContent = []
            tmpLabel = []
            for i in range(len(key)):
                tmpContent.extend(key[i])
            content.append(tmpContent)
            for i in range(len(self.trainSet[key])):
                tmpLabel.extend(self.trainSet[key][i])
            label.append(tmpLabel)
        self.data = []  # 训练数据
        for i in range(len(label)):
            self.data.append((content[i], label[i]))
        self.word_to_ix = {}  # 字典
        for sentence, tags in self.data:
            for word in sentence:  # word为句子中的每个字
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)  # 如果这个word不在字典中，那么将它加入字典，并且对应当前字典的长度
        self.tagset_size = len(self.tag_to_ix)

        # 网络结构
        self.word_embeds = nn.Embedding(len(self.word_to_ix), self.embedding_dim)  # 字典的大小和词嵌入的维度
        self.hidden = self.init_hidden()
        # LSTM以word_embeddings作为输入, 输出维度为 hidden_dim 的隐藏状态值
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True)
        # 线性层将隐藏状态空间映射到标注空间
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

        # 转换参数矩阵。 输入i，j是得分从j转换到i。
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        # 这两个语句强制执行我们从不转移到开始标记的约束，并且我们永远不会从停止标记转移
        self.transitions.data[self.tag_to_ix[self.START_TAG], :] = -10000
        self.transitions.data[:, self.tag_to_ix[self.STOP_TAG]] = -10000

    def prepare_sequence(selr, seq, to_ix):  # seq是句子的字序列，idxs是字序列对应的向量，to_ix是字和序号的字典
        idxs = [to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)

    def argMax(self, vec):
        # 将argmax作为python int返回
        _, idx = torch.max(vec, 1)
        return idx.item()

    # 以正向算法的数值稳定方式计算log sum exp
    def log_sum_exp(self, vec):
        max_score = vec[0, self.argMax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + \
               torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

    # 计算前向计算后的feat得到的score
    def forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG包含所有得分
        init_alphas[0][self.tag_to_ix[self.START_TAG]] = 0.
        # 包装一个变量，以便我们获得自动反向提升
        forward_var = init_alphas
        # 通过句子迭代
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # 广播发射得分：无论以前的标记是怎样的都是相同的
                trans_score = self.transitions[next_tag].view(1, -1)
                # trans_score的第i个条目是从i转换到next_tag的分数
                next_tag_var = forward_var + trans_score + emit_score
                # 此标记的转发变量是所有分数的log-sum-exp。
                alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        alpha = self.log_sum_exp(terminal_var)
        return alpha

    def init_hidden(self):
        # 一开始并没有隐藏状态所以我们要先初始化一个
        # 各个维度的含义是 (num_layers, minibatch_size, hidden_dim)
        return torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2)

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    # 计算gold score
    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[self.START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[self.STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []
        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[self.START_TAG]] = 0
        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step
            for next_tag in range(self.tagset_size):
                # next_tag_var [i]保存上一步的标签i的维特比变量
                # 加上从标签i转换到next_tag的分数。
                # 我们这里不包括emission分数，因为最大值不依赖于它们（我们在下面添加它们）
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = self.argMax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # 现在添加emission分数，并将forward_var分配给我们刚刚计算的维特比变量集
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
        # 过渡到STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        best_tag_id = self.argMax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
        # 按照后退指针解码最佳路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # 弹出开始标记
        start = best_path.pop()
        assert start == self.tag_to_ix[self.START_TAG]
        best_path.reverse()
        return path_score, best_path

    def loss(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self.forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        # 获取BiLSTM的emission分数
        lstm_feats = self._get_lstm_features(sentence)
        # 找到最佳路径
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

    def myTrain(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        # optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=1e-4)
        for epoch in range(self.epochs):
            print("epoch " + str(epoch + 1) + " 开始")
            for sentence, tags in self.data:
                self.zero_grad()
                sentence_in = self.prepare_sequence(sentence, self.word_to_ix)
                targets = torch.tensor([self.tag_to_ix[t] for t in tags], dtype=torch.long)
                loss = self.loss(sentence_in, targets)
                loss.backward()
                optimizer.step()
            print('epoch/epochs:{}/{},loss:{:.6f}'.format(epoch + 1, self.epochs, loss.data[0]))
            torch.save(self, '../models/embedding/lstm-adam-epoch=' + str(epoch + 1) + '.model')

    def predict(self, sentence):
        # print(sentence)
        for i in range(len(sentence)):
            if sentence[i] not in list(self.word_to_ix.keys()):
                tmp = list(sentence)
                tmp[i] = '黜'  # 频率最低词
                sentence = ''.join(tmp)
        # print(sentence)
        net = torch.load('models/embedding/lstm-adam-epoch=14.model')
        net.eval()
        precheck_sent = self.prepare_sequence(sentence, self.word_to_ix)
        label = net(precheck_sent)[1]  # 调用forward
        str = ""
        for i in label:
            if i == 0:
                str += 'B'
            elif i == 1:
                str += 'I'
            elif i == 2:
                str += 'E'
            elif i == 3:
                str += 'S'
        return str


if __name__ == '__main__':
    model = BiLSTM_CRF()
    # model.myTrain()
    # print(model.predict("丰子恺当年送他三句话：“多读书，广结交，少说话”——－（附图片１张）"))
