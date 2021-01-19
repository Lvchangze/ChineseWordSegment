from wordseg.solution import Solution
from wordseg.hmm_model import HMMModel
from wordseg.crf_model import CRFModel
from wordseg.lstm_word2vec import BiLSTM_CRF  # word2vec
# from wordseg.lstm_embedding import BiLSTM_CRF  # embedding
import wordseg.base_line_model

solution = Solution()

funcs = {
    "HMM": solution.hmm_predict,
    "CRF": solution.crf_predict,
    "BiLSTM-CRF": solution.dnn_predict,
    "jieba": solution.jieba_predict,
    "snownlp": solution.snownlp_predict,
    "pkuseg": solution.pkuseg_predict,
    "thulac": solution.thulac_predict,
    "pyhanlp": solution.pyhanlp_predict
}

# inputs = ["我爱北京天安门", "今天天气怎么样", "中华人民共和国", "腾讯是个好公司"]
#
# for f_name, func in funcs.items():
#     outputs = func(inputs)
#     print(f"对于输入{inputs}，你的输出是{outputs}。")

for f_name, func in funcs.items():
    print(f"\n{f_name}模型 -->")
    examples = open("test_dataset/input.utf8", encoding="utf8").readlines()
    examples = [ele.strip() for ele in examples]
    gold = open("test_dataset/gold.utf8", encoding="utf8").readlines()
    gold = [ele.strip() for ele in gold]
    pred = func(examples)
    accuracys = []
    if pred is not None:
        for i in range(len(examples)):
            corr = [1 if a == b else 0 for a, b in zip(str(gold[i]), str(pred[i]))]
            accu = sum(corr) / len(corr)
            accuracys.append(accu)
        print(f"准确率：{sum(accuracys) / len(accuracys)}")
    else:
        print(f"{f_name}暂无结果")
