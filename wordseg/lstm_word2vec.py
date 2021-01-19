import torch
import torch.nn as nn
import torch.optim as optim

# 初始化
torch.manual_seed(1)


def argmax(vec):
    idx = torch.argmax(vec)
    return idx.item()


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


# 向量库类
class VectorLibrary:
    def __init__(self, filename, encoding='utf8', device=torch.device('cpu')):
        self.data = dict()
        self.device = device
        with open(filename, 'r', encoding=encoding) as f:
            rows = f.readlines()
            self.vocab_size, self.vector_dim = tuple(map(int, rows[0].split()))
            for row in rows[1:]:
                arr = row.split()
                word = arr[0]
                vector = torch.tensor(list(map(float, arr[1:])), device=device)
                self.data[word] = vector

    def __getitem__(self, key):
        if key in self.data.keys():
            return self.data[key]
        else:  # 处理为未登陆词，随机赋值
            self.data[key] = torch.randn(self.vector_dim, device=self.device)
            return self.data[key]

    def __len__(self):
        return len(self.data)


class TrainDataSet:
    def __init__(self, filename, vector_library, encoding='utf8'):
        self.tag_to_ix = {}
        self.data = []
        self.device = vector_library.device
        self.vector_library = vector_library

        with open(filename, 'r', encoding=encoding) as f:
            rows = f.readlines()
            sentence = ([], [])
            for row in rows:
                i = row.strip()
                if len(i) > 0:
                    tag = 'NIL'
                    i = i.split()
                    if len(i) == 1:
                        sentence[0].append(i[0])
                        sentence[1].append(tag)
                    else:
                        word, tag = i
                        sentence[0].append(word)
                        sentence[1].append(tag)

                    if tag not in self.tag_to_ix:
                        self.tag_to_ix[tag] = len(self.tag_to_ix)

                else:
                    self.data.append(sentence)
                    sentence = ([], [])

            if len(sentence) > 0:
                self.data.append(sentence)

    def prepareWordSequence(self, seq):
        return [self.vector_library[w] for w in seq]

    def prepareTagSequence(self, seq):
        idxs = [self.tag_to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long, device=self.device)

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)


class BiLSTM_CRF(nn.Module):
    def __init__(self, input_dim, hidden_dim, tag_to_ix, device=torch.device('cpu')):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.tag_to_ix = tag_to_ix

        # 增加开始标签、结束标签
        self.END_TAG = '<END_TAG>'
        self.START_TAG = '<START_TAG>'
        self.tag_to_ix[self.END_TAG] = len(self.tag_to_ix)
        self.tag_to_ix[self.START_TAG] = len(self.tag_to_ix)

        self.END_TAG = self.tag_to_ix[self.END_TAG]
        self.START_TAG = self.tag_to_ix[self.START_TAG]

        self.tagset_size = len(self.tag_to_ix)

        # 初始化网络参数+
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[self.START_TAG, :] = -10000
        self.transitions.data[:, self.END_TAG] = -10000
        self.hidden = self.initHidden()

        self.to(self.device)

    def initHidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2, device=self.device),
                torch.randn(2, 1, self.hidden_dim // 2, device=self.device))

    def get_lstm_features(self, sequence):
        size = len(sequence)
        self.hidden = self.initHidden()
        embeds = torch.cat(sequence).view(size, 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(size, self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def Viterbi(self, feats):
        backpointers = []

        init_vvars = torch.full((1, self.tagset_size), -10000., device=self.device)
        init_vvars[0][self.START_TAG] = 0

        forward_vars = init_vvars

        for feat in feats:
            bptrs = []
            viterbivars = []

            for next_tag in range(self.tagset_size):
                next_tag_vars = forward_vars + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_vars)
                bptrs.append(best_tag_id)
                viterbivars.append(next_tag_vars[0][best_tag_id].view(1))

            forward_vars = (torch.cat(viterbivars) + feat).view(1, -1)
            backpointers.append(bptrs)

        # 以 <END_TAG> 为结尾
        terminal_var = forward_vars + self.transitions[self.END_TAG]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]

        for bptrs in reversed(backpointers):
            best_tag_id = bptrs[best_tag_id]
            best_path.append(best_tag_id)

        # 以 <START_TAG> 为开头
        start = best_path.pop()
        assert start == self.START_TAG

        best_path.reverse()

        return path_score, best_path

    def forward(self, sequence):
        lstm_feats = self.get_lstm_features(sequence)
        score, best_seq = self.Viterbi(lstm_feats)
        return score, best_seq

    def loss(self, sequence, tags):
        lstm_feats = self.get_lstm_features(sequence)
        forward_score = self.forward_alg(lstm_feats)
        gold_score = self.get_sentence_score(lstm_feats, tags)
        return forward_score - gold_score  # 负

    def get_sentence_score(self, feats, tags):
        score = torch.zeros(1, device=self.device)
        tags = torch.cat([torch.tensor([self.START_TAG], dtype=torch.long, device=self.device), tags])
        for (i, feat) in enumerate(feats):
            emit_score = feat[tags[i + 1]]
            trans_score = self.transitions[tags[i + 1], tags[i]]
            score = score + emit_score + trans_score
        score = score + self.transitions[self.END_TAG, tags[-1]]
        return score

    def forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000., device=self.device)
        init_alphas[0][self.START_TAG] = 0.

        forward_vars = init_alphas
        for feat in feats:
            alphas = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_vars + trans_score + emit_score
                alphas.append(log_sum_exp(next_tag_var).view(1))
            forward_vars = torch.cat(alphas).view(1, -1)
        terminal_var = forward_vars + self.transitions[self.END_TAG]
        return log_sum_exp(terminal_var)


def train():
    epoch = 100
    device = torch.device('cpu')
    vl = VectorLibrary('vector_library.utf8', device=device)
    training_set = TrainDataSet('../dataset/dataset1/train.utf8', vl)
    model = BiLSTM_CRF(vl.vector_dim, 150, training_set.tag_to_ix, device)  # 创建 BiLSTM-CRF 网络
    torch.device('cpu')
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=1e-4)
    for e in range(epoch):
        print("epoch " + str(e + 1) + " 开始")
        for (i, (sentence, tags)) in enumerate(training_set):
            model.zero_grad()
            sentence_in = training_set.prepareWordSequence(sentence)
            tagSets = training_set.prepareTagSequence(tags)
            loss = model.loss(sentence_in, tagSets)
            loss.backward()
            optimizer.step()
        print('epoch/epochs:{}/{},loss:{:.6f}'.format(e + 1, epoch, loss.data[0]))
        torch.save(model, '../models/lstm-epoch' + str(e + 1) + '.model')


count = 0


def predict(sentence, net, vector_library):
    global count
    count += 1
    # print(count)
    # vector_library = VectorLibrary('wordseg/vector_library.utf8')
    sentence_in = [vector_library[w] for w in sentence]
    _, label = net(sentence_in)
    str = ""
    for i in label:
        if i == 0:
            str += 'B'
        elif i == 1:
            str += 'E'
        elif i == 2:
            str += 'I'
        elif i == 3:
            str += 'S'
    return str


if __name__ == '__main__':
    train()
    # print(predict("丰子恺当年送他三句话：“多读书，广结交，少说话”——－（附图片１张）"))
