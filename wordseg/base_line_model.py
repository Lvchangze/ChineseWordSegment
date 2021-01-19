import jieba
from snownlp import SnowNLP
from pyhanlp import HanLP


def jieba_predict(sentence):
    seg_list = jieba.cut(sentence)
    my = list(seg_list)
    result = ""
    for i in my:
        if len(i) == 1:
            result += "S"
        elif len(i) == 2:
            result += "BE"
        else:
            iNum = len(i) - 2
            result += "B"
            while iNum > 0:
                result += "I"
                iNum -= 1
            result += "E"
    return result


def snownlp_predict(sentence):
    s = SnowNLP(sentence)
    my = list(s.words)
    result = ""
    for i in my:
        if len(i) == 1:
            result += "S"
        elif len(i) == 2:
            result += "BE"
        else:
            iNum = len(i) - 2
            result += "B"
            while iNum > 0:
                result += "I"
                iNum -= 1
            result += "E"
    return result


def pkuseg_predict(sentence, pku_seg):
    my = pku_seg.cut(sentence)
    result = ""
    for i in my:
        if len(i) == 1:
            result += "S"
        elif len(i) == 2:
            result += "BE"
        else:
            iNum = len(i) - 2
            result += "B"
            while iNum > 0:
                result += "I"
                iNum -= 1
            result += "E"
    return result


def thulac_predict(sentence, thu_lac):
    thu_result = thu_lac.cut(sentence, text=True)
    my = thu_result.split(" ")
    result = ""
    for i in my:
        if i == " ":
            continue
        elif len(i) == 1:
            result += "S"
        elif len(i) == 2:
            result += "BE"
        else:
            iNum = len(i) - 2
            result += "B"
            while iNum > 0:
                result += "I"
                iNum -= 1
            result += "E"
    return result


def pyhanlp_predict(sentence):
    han_word_seg = HanLP.segment(sentence)
    my = (' '.join([term.word for term in han_word_seg])).split(" ")
    result = ""
    for i in my:
        if len(i) == 1:
            result += "S"
        elif len(i) == 2:
            result += "BE"
        else:
            iNum = len(i) - 2
            result += "B"
            while iNum > 0:
                result += "I"
                iNum -= 1
            result += "E"
    return result

