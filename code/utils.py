#! /usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
from nltk.stem import SnowballStemmer
import pickle
import sys
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import configparser as ConfigParser
try:
    import lzma
except:
    pass

class MyFunction(object):
    def read_embedding(filename):
        embed = {}
        for line in open(filename,encoding='utf-8'):
            line = line.strip().split()
            embed[str(line[0])] = list(map(float, line[1:]))
        print('[%s]\n\tEmbedding size: %d' % (filename, len(embed)), end='\n')
        return embed


class Loader:
    def __init__(self, config_fp):
        self.feature_name = self.__class__.__name__
        self.config = ConfigParser.ConfigParser()
        self.config.read(config_fp)

    def loadFeatures(self, DataSpanishSenPair, features_name):
        Features = pd.DataFrame()
        for cur_featureName in features_name:
            path = '../data/Feature/' + cur_featureName + '.smat'
            features_file = open(path)
            line = features_file.readline()
            numsamle, numSubFeatures = [int(e) for e in line.split()]
            line = features_file.readline()
            cur_feature = np.zeros((numsamle, numSubFeatures))
            count = 0
            while line:
                cur_feature[count, :] = [e.split(":")[1] for e in line.split()]
                line = features_file.readline()
                count += 1
            for i in range(numSubFeatures):
                cur_name = cur_featureName + '_' + str(i)
                Features[cur_name] = cur_feature[:, i]
        DataSpanishSenPair['Features'] = Features

    def loadLabel(self, DataSpanishSenPair):
        path = '../data/Feature/Label_0.smat'
        features_file = open(path)
        line = features_file.readline()
        numsamle, numSubFeatures = [int(e) for e in line.split()]
        line = features_file.readline()
        Labels_list = np.zeros((numsamle, numSubFeatures))
        count = 0
        while line:
            Labels_list[count] = [e.split(":")[1] for e in line.split()]
            line = features_file.readline()
            count += 1
        DataSpanishSenPair['Labels'] = Labels_list

    def loadAllData(self):
        feature_pt = self.config.get('FEATURE', 'feature_selected')
        features_name = feature_pt.split()
        num_features = len(features_name)
        DataSpanishSenPair = {}
        DataSpanishSenPair['featuresName'] = features_name
        DataSpanishSenPair['LabelName'] = 'isSameMeaning'
        DataSpanishSenPair['NumberFeatures'] = num_features
        self.loadFeatures(DataSpanishSenPair, features_name)
        self.loadLabel(DataSpanishSenPair)
        return DataSpanishSenPair


class Processing(object):
    def __init__(self):
        pass

    def excute(self, data_pt, save_pt):
        data = File.readVector(data_pt)
        spanish_sentence1 = []
        spanish_sentence2 = []
        is_duplicateline = []

        for row in data:
            processed_spa1 = self.removePunctuationAndLower(row.split('\t')[0])
            processed_spa1 = self.stemming(processed_spa1)
            processed_spa2 = self.removePunctuationAndLower(row.split('\t')[2])
            processed_spa2 = self.stemming(processed_spa2)
            spanish_sentence1.append(processed_spa1)
            spanish_sentence2.append(processed_spa2)
            is_duplicateline.append(row.split('\t')[4])
        data_frame = pd.DataFrame({'spanish_sentence1':spanish_sentence1, 'spanish_sentence2':spanish_sentence2, 'is_duplicate':is_duplicateline})
        data_frame.to_csv(save_pt, index=False, encoding='UTF-8')

    def stemming(self, row):
        _stemmer = SnowballStemmer('spanish')
        line = [_stemmer.stem(word) for word in row.split()]
        return ' '.join(line)

    def removePunctuationAndLower(self, row):
        r = '¿[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+?'
        line = re.sub(r, '', row)
        line = line.strip('¿?')
        return line.lower()


class File:
    @staticmethod
    def readVector(fp):
        f = open(fp, 'r', encoding='UTF-8')
        result = []
        line = f.readline()
        while line:
            result.append(line.strip())
            line = f.readline()
        return result


class LogUtil(object):
    def __init__(self):
        pass

    @staticmethod
    def log(typ, msg):
        print("[%s]\t[%s]\t%s" % (TimeUtil.t_now(), typ, str(msg)))
        sys.stdout.flush()
        return


class NgramUtil(object):

    def __init__(self):
        pass

    @staticmethod
    def unigrams(words):
        """
            Input: a list of words, e.g., ["I", "am", "Denny"]
            Output: a list of unigram
        """
        assert type(words) == list
        return words

    @staticmethod
    def bigrams(words, join_string, skip=0):
        """
           Input: a list of words, e.g., ["I", "am", "Denny"]
           Output: a list of bigram, e.g., ["I_am", "am_Denny"]
        """
        assert type(words) == list
        L = len(words)
        if L > 1:
            lst = []
            for i in range(L - 1):
                for k in range(1, skip + 2):
                    if i + k < L:
                        lst.append(join_string.join([str(words[i]), str(words[i + k])]))
        else:
            # set it as unigram
            lst = NgramUtil.unigrams(words)
        return lst

    @staticmethod
    def trigrams(words, join_string, skip=0):
        """
           Input: a list of words, e.g., ["I", "am", "Denny"]
           Output: a list of trigram, e.g., ["I_am_Denny"]
        """
        assert type(words) == list
        L = len(words)
        if L > 2:
            lst = []
            for i in range(L - 2):
                for k1 in range(1, skip + 2):
                    for k2 in range(1, skip + 2):
                        if i + k1 < L and i + k1 + k2 < L:
                            lst.append(join_string.join([str(words[i]), str(words[i + k1]), str(words[i + k1 + k2])]))
        else:
            # set it as bigram
            lst = NgramUtil.bigrams(words, join_string, skip)
        return lst

    @staticmethod
    def fourgrams(words, join_string):
        """
            Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
            Output: a list of trigram, e.g., ["I_am_Denny_boy"]
        """
        assert type(words) == list
        L = len(words)
        if L > 3:
            lst = []
            for i in xrange(L - 3):
                lst.append(join_string.join([words[i], words[i + 1], words[i + 2], words[i + 3]]))
        else:
            # set it as trigram
            lst = NgramUtil.trigrams(words, join_string)
        return lst

    @staticmethod
    def uniterms(words):
        return NgramUtil.unigrams(words)

    @staticmethod
    def biterms(words, join_string):
        """
            Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
            Output: a list of biterm, e.g., ["I_am", "I_Denny", "I_boy", "am_Denny", "am_boy", "Denny_boy"]
        """
        assert type(words) == list
        L = len(words)
        if L > 1:
            lst = []
            for i in range(L - 1):
                for j in range(i + 1, L):
                    lst.append(join_string.join([words[i], words[j]]))
        else:
            # set it as uniterm
            lst = NgramUtil.uniterms(words)
        return lst

    @staticmethod
    def triterms(words, join_string):
        """
            Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
            Output: a list of triterm, e.g., ["I_am_Denny", "I_am_boy", "I_Denny_boy", "am_Denny_boy"]
        """
        assert type(words) == list
        L = len(words)
        if L > 2:
            lst = []
            for i in xrange(L - 2):
                for j in xrange(i + 1, L - 1):
                    for k in xrange(j + 1, L):
                        lst.append(join_string.join([words[i], words[j], words[k]]))
        else:
            # set it as biterm
            lst = NgramUtil.biterms(words, join_string)
        return lst

    @staticmethod
    def fourterms(words, join_string):
        """
            Input: a list of words, e.g., ["I", "am", "Denny", "boy", "ha"]
            Output: a list of fourterm, e.g., ["I_am_Denny_boy", "I_am_Denny_ha", "I_am_boy_ha", "I_Denny_boy_ha", "am_Denny_boy_ha"]
        """
        assert type(words) == list
        L = len(words)
        if L > 3:
            lst = []
            for i in xrange(L - 3):
                for j in xrange(i + 1, L - 2):
                    for k in xrange(j + 1, L - 1):
                        for l in xrange(k + 1, L):
                            lst.append(join_string.join([words[i], words[j], words[k], words[l]]))
        else:
            # set it as triterm
            lst = NgramUtil.triterms(words, join_string)
        return lst

    @staticmethod
    def ngrams(words, ngram, join_string=" "):
        """
        wrapper for ngram
        """
        if ngram == 1:
            return NgramUtil.unigrams(words)
        elif ngram == 2:
            return NgramUtil.bigrams(words, join_string)
        elif ngram == 3:
            return NgramUtil.trigrams(words, join_string)
        elif ngram == 4:
            return NgramUtil.fourgrams(words, join_string)
        elif ngram == 12:
            unigram = NgramUtil.unigrams(words)
            bigram = [x for x in NgramUtil.bigrams(words, join_string) if len(x.split(join_string)) == 2]
            return unigram + bigram
        elif ngram == 123:
            unigram = NgramUtil.unigrams(words)
            bigram = [x for x in NgramUtil.bigrams(words, join_string) if len(x.split(join_string)) == 2]
            trigram = [x for x in NgramUtil.trigrams(words, join_string) if len(x.split(join_string)) == 3]
            return unigram + bigram + trigram

    @staticmethod
    def nterms(words, nterm, join_string=" "):
        """wrapper for nterm"""
        if nterm == 1:
            return NgramUtil.uniterms(words)
        elif nterm == 2:
            return NgramUtil.biterms(words, join_string)
        elif nterm == 3:
            return NgramUtil.triterms(words, join_string)
        elif nterm == 4:
            return NgramUtil.fourterms(words, join_string)


class DistanceUtil(object):
    """
    Tool of Distance
    """

    @staticmethod
    def edit_dist(str1, str2):
        try:
            # very fast
            # http://stackoverflow.com/questions/14260126/how-python-levenshtein-ratio-is-computed
            import Levenshtein
            d = Levenshtein.distance(str1, str2) / float(max(len(str1), len(str2)))
        except:
            # https://docs.python.org/2/library/difflib.html
            d = 1. - SequenceMatcher(lambda x: x == " ", str1, str2).ratio()
        return d

    @staticmethod
    def n_gram_over_lap(sen1, sen2, n):
        len1 = len(sen1)
        len2 = len(sen2)

        word_set1 = set()
        word_set2 = set()

        for i in range(len1):
            if i <= n-2:
                continue
            join_str = ""
            for j in range(n-1, -1, -1):
                join_str += sen1[i-j]
            word_set1.add(join_str)

        for i in range(len2):
            if i <= n-2:
                continue
            join_str = ""
            for j in range(n-1, -1, -1):
                join_str += sen2[i-j]
            word_set2.add(join_str)
        num1 = len(word_set1 & word_set2)
        num2 = len(word_set1) + len(word_set2)
        if num2 == 0:
            return 0
        return num1 * 1.0 / num2

    @staticmethod
    def is_str_match(str1, str2, threshold=1.0):
        assert threshold >= 0.0 and threshold <= 1.0, "Wrong threshold."
        if float(threshold) == 1.0:
            return str1 == str2
        else:
            return (1. - DistanceUtil.edit_dist(str1, str2)) >= threshold

    @staticmethod
    def longest_match_size(str1, str2):
        sq = SequenceMatcher(lambda x: x == " ", str1, str2)
        match = sq.find_longest_match(0, len(str1), 0, len(str2))
        return match.size

    @staticmethod
    def longest_match_ratio(str1, str2):
        sq = SequenceMatcher(lambda x: x == " ", str1, str2)
        match = sq.find_longest_match(0, len(str1), 0, len(str2))
        return MathUtil.try_divide(match.size, min(len(str1), len(str2)))

    @staticmethod
    def longest_match_size(str1, str2):
        sq = SequenceMatcher(lambda x: x == " ", str1, str2)
        match = sq.find_longest_match(0, len(str1), 0, len(str2))
        return match.size

    @staticmethod
    def longest_match_ratio(str1, str2):
        sq = SequenceMatcher(lambda x: x == " ", str1, str2)
        match = sq.find_longest_match(0, len(str1), 0, len(str2))
        return MathUtil.try_divide(match.size, min(len(str1), len(str2)))

    @staticmethod
    def compression_dist(x, y, l_x=None, l_y=None):
        if x == y:
            return 0
        x_b = x.encode('utf-8')
        y_b = y.encode('utf-8')
        if l_x is None:
            l_x = len(lzma.compress(x_b))
            l_y = len(lzma.compress(y_b))
        l_xy = len(lzma.compress(x_b + y_b))
        l_yx = len(lzma.compress(y_b + x_b))
        dist = MathUtil.try_divide(min(l_xy, l_yx) - min(l_x, l_y), max(l_x, l_y))
        return dist

    @staticmethod
    def cosine_sim(vec1, vec2):
        try:
            s = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
        except:
            try:
                s = cosine_similarity(vec1, vec2)[0][0]
            except:
                s = MISSING_VALUE_NUMERIC
        return s

    @staticmethod
    def jaccard_coef(A, B):
        if not isinstance(A, set):
            A = set(A)
        if not isinstance(B, set):
            B = set(B)
        return MathUtil.try_divide(float(len(A.intersection(B))), len(A.union(B)))

    @staticmethod
    def dice_dist(A, B):
        if not isinstance(A, set):
            A = set(A)
        if not isinstance(B, set):
            B = set(B)
        return MathUtil.try_divide(2. * float(len(A.intersection(B))), (len(A) + len(B)))


class MathUtil(object):
    """
    Tool of Math
    """

    @staticmethod
    def count_one_bits(x):
        """
        Calculate the number of bits which are 1
        :param x: number which will be calculated
        :return: number of bits in `x`
        """
        n = 0
        while x:
            n += 1 if (x & 0x01) else 0
            x >>= 1
        return n

    @staticmethod
    def int2binarystr(x):
        """
        Convert the number from decimal to binary
        :param x: decimal number
        :return: string represented binary format of `x`
        """
        s = ""
        while x:
            s += "1" if (x & 0x01) else "0"
            x >>= 1
        return s[::-1]

    @staticmethod
    def try_divide(x, y, val=0.0):
        """
        try to divide two numbers
        """
        if y != 0.0:
            val = float(x) / y
        return val

    @staticmethod
    def corr(x, y_train):
        """
        Calculate correlation between specified feature and labels
        :param x: specified feature in numpy
        :param y_train: labels in numpy
        :return: value of correlation
        """
        if MathUtil.dim(x) == 1:
            corr = pearsonr(x.flatten(), y_train)[0]
            if str(corr) == "nan":
                corr = 0.
        else:
            corr = 1.
        return corr

    @staticmethod
    def dim(x):
        d = 1 if len(x.shape) == 1 else x.shape[1]
        return d

    @staticmethod
    def aggregate(data, modes):
        valid_modes = ["size", "mean", "std", "max", "min", "median"]

        if isinstance(modes, str):
            assert modes.lower() in valid_modes, "Wrong aggregation_mode: %s" % modes
            modes = [modes.lower()]
        elif isinstance(modes, list):
            for m in modes:
                assert m.lower() in valid_modes, "Wrong aggregation_mode: %s" % m
                modes = [m.lower() for m in modes]
        aggregators = [getattr(np, m) for m in modes]

        aggeration_value = list()
        for agg in aggregators:
            try:
                s = agg(data)
            except ValueError:
                s = MISSING_VALUE_NUMERIC
            aggeration_value.append(s)
        return aggeration_value

    @staticmethod
    def cut_prob(p):
        p[p > 1.0 - 1e-15] = 1.0 - 1e-15
        p[p < 1e-15] = 1e-15
        return p

    @staticmethod
    def logit(p):
        assert isinstance(p, np.ndarray), 'type error'
        p = MathUtil.cut_prob(p)
        return np.log(p / (1. - p))

    @staticmethod
    def logistic(y):
        assert isinstance(p, np.ndarray), 'type error'
        return np.exp(y) / (1. + np.exp(y))
