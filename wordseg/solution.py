from typing import List
import torch


class Solution:

    def hmm_predict(self, sentences: List[str]) -> List[str]:
        from .hmm_model import predict
        model = torch.load('models/hmm.model')
        results = []
        for sent in sentences:
            # print(predict(sent))
            results.append(predict(sent, model))
        return results

    def crf_predict(self, sentences: List[str]) -> List[str]:
        from .crf_model import CRFModel
        model = CRFModel()
        checkpoint = torch.load("models/CRF-dataSet2-备选.model")
        results = []
        for sent in sentences:
            # print(model.predict(sent))
            results.append(model.predict(sent, checkpoint))
        return results

    def dnn_predict(self, sentences: List[str]) -> List[str]:
        from .lstm_word2vec import predict
        from .lstm_word2vec import VectorLibrary
        net = torch.load('models/word2vec/lstm-sgd-epoch12.model')
        vector_library = VectorLibrary('wordseg/vector_library.utf8')
        results = []
        for sent in sentences:
            results.append(predict(sent, net, vector_library))
        return results

        # from .lstm_embedding import BiLSTM_CRF
        # model = BiLSTM_CRF()
        # results = []
        # for sent in sentences:
        #     results.append(model.predict(sent))
        # return results

    def jieba_predict(self, sentences):
        from .base_line_model import jieba_predict
        results = []
        for sent in sentences:
            results.append(jieba_predict(sent))
        return results

    def snownlp_predict(self, sentences):
        from .base_line_model import snownlp_predict
        results = []
        for sent in sentences:
            results.append(snownlp_predict(sent))
        return results

    def pkuseg_predict(self, sentences):
        from .base_line_model import pkuseg_predict
        import pkuseg
        pku_seg = pkuseg.pkuseg()
        results = []
        for sent in sentences:
            results.append(pkuseg_predict(sent, pku_seg))
        return results

    def thulac_predict(self, sentences):
        from .base_line_model import thulac_predict
        import thulac
        thu_lac = thulac.thulac(seg_only=True)
        results = []
        for sent in sentences:
            results.append(thulac_predict(sent, thu_lac))
        return results

    def pyhanlp_predict(self, sentences):
        from .base_line_model import pyhanlp_predict
        results = []
        for sent in sentences:
            results.append(pyhanlp_predict(sent))
        return results