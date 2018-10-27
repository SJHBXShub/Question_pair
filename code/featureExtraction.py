import configparser as ConfigParser
import pandas as pd
import sys
from imp import reload
import math
import chardet
import nltk
import numpy as np
import pandas as pd
import configparser as ConfigParser
from nltk.stem import SnowballStemmer
from feature import Feature
from nltk.corpus import stopwords
from utils import NgramUtil, DistanceUtil
from fuzzywuzzy import fuzz

stops = set(stopwords.words("spanish"))
snowball_stemmer = SnowballStemmer('spanish')


class Extractor(object):
    def __init__(self, config_fp):
        self.feature_name = self.__class__.__name__
        self.config = ConfigParser.ConfigParser()
        self.config.read(config_fp)

    def get_feature_num(self):
        assert False, 'Please override function: Extractor.get_feature_num()'

    def extract_row(self, row):
        assert False, 'Please override function: Extractor.extract_row()'

    def extract(self, data_fp, feature_version):
        feature_pt = self.config.get('DIRECTORY', 'feature_pt')
        feature_fp = '%s/%s_%s.smat' % (feature_pt, self.feature_name, feature_version)

        data = pd.read_csv(data_fp, encoding='UTF-8').fillna(value="")
        num_sample = len(data)
        num_chile_features = int(self.get_feature_num())
        feature_file = open(feature_fp, 'w')
        feature_file.write('%d %d\n' % (num_sample, num_chile_features))
        for index, row in data.iterrows():
            feature = self.extract_row(row)
            if index == 0:
                print(row)
                print(feature)
            Feature.checkFeature(feature)
            Feature.save_feature(feature, feature_file)
        feature_file.close()


class Not(Extractor):
    def __init__(self, config_fp):
        Extractor.__init__(self, config_fp)
        self.snowball_stemmer = SnowballStemmer('spanish')

    def get_feature_num(self):
        return 5

    def extract_row(self, row):
        q1 = str(row['spanish_sentence1']).strip()
        q2 = str(row['spanish_sentence2']).strip()

        q1_words = [self.snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(q1)]
        q2_words = [self.snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(q2)]

        not_cnt1 = q1_words.count(b'no')
        not_cnt2 = q2_words.count(b'no')
        not_cnt1 += q1_words.count(b'ni')
        not_cnt2 += q2_words.count(b'ni')
        not_cnt1 += q1_words.count(b'nunca')
        not_cnt2 += q2_words.count(b'nunca')

        fs = list()
        fs.append(not_cnt1)
        fs.append(not_cnt2)
        if not_cnt1 > 0 and not_cnt2 > 0:
            fs.append(1.)
        else:
            fs.append(0.)
        if (not_cnt1 > 0) or (not_cnt2 > 0):
            fs.append(1.)
        else:
            fs.append(0.)
        if not_cnt2 <= 0 < not_cnt1 or not_cnt1 <= 0 < not_cnt2:
            fs.append(1.)
        else:
            fs.append(0.)

        return fs


class WordMatchShare(Extractor):

    def extract_row(self, row):
        q1words = {}
        q2words = {}
        for word in str(row['spanish_sentence1']).lower().split():
            if word not in stops:
                q1words[word] = q1words.get(word, 0) + 1
        for word in str(row['spanish_sentence2']).lower().split():
            if word not in stops:
                q2words[word] = q2words.get(word, 0) + 1
        n_shared_word_in_q1 = sum([q1words[w] for w in q1words if w in q2words])
        n_shared_word_in_q2 = sum([q2words[w] for w in q2words if w in q1words])
        n_tol = sum(q1words.values()) + sum(q2words.values())
        if 1e-6 > n_tol:
            return [0.]
        else:
            return [round(1.0 * (n_shared_word_in_q1 + n_shared_word_in_q2) / n_tol,4)]

    def get_feature_num(self):
        return 1


class Length(Extractor):
    def extract_row(self, row):
        q1 = str(row['spanish_sentence1'])
        q2 = str(row['spanish_sentence2'])

        fs = list()
        fs.append(len(q1))
        fs.append(len(q2))
        fs.append(len(q1.split()))
        fs.append(len(q2.split()))
        return fs

    def get_feature_num(self):
        return 4


class NgramJaccardCoef(Extractor):
    def extract_row(self, row):
        q1_words = [snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(str(row['spanish_sentence1']))]
        q2_words = [snowball_stemmer.stem(word).encode('utf-8') for word in
                    nltk.word_tokenize(str(row['spanish_sentence2']))]
        fs = list()
        for n in range(1, 4):
            q1_ngrams = NgramUtil.ngrams(q1_words, n)
            q2_ngrams = NgramUtil.ngrams(q2_words, n)
            fs.append(DistanceUtil.jaccard_coef(q1_ngrams, q2_ngrams))
        return fs

    def get_feature_num(self):
        return 3


class long_common_sequence(Extractor):
    def extract_row(self, row):
        q1_words = (str(row['spanish_sentence1']).lower().split())
        q2_words = (str(row['spanish_sentence2']).lower().split())
        len1 = len(q1_words)
        len2 = len(q2_words)
        dp = [ [ 0 for j in range(len2) ] for i in range(len1) ]
        for i in range(len1):
            for j in range(len2):
                if q1_words[i] == q2_words[j]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return [dp[len1-1][len2-1] * 1.0 / (len1 + len2)]

    def get_feature_num(self):
        return 1


class fuzz_partial_token_set_ratio(Extractor):
    def extract_row(self, row):
        q1_words = set(str(row['spanish_sentence1']).lower().split())
        q2_words = set(str(row['spanish_sentence2']).lower().split())
        return [fuzz.partial_token_set_ratio(q1_words, q2_words)]

    def get_feature_num(self):
        return 1


class fuzz_partial_token_sort_ratio(Extractor):
    def extract_row(self, row):
        q1_words = set(str(row['spanish_sentence1']).lower().split())
        q2_words = set(str(row['spanish_sentence2']).lower().split())
        return [fuzz.partial_token_sort_ratio(q1_words, q2_words)]

    def get_feature_num(self):
        return 1


class TFIDFSpanish(Extractor):
    def __init__(self, config, data_fp):
        Extractor.__init__(self, config)
        train_data = pd.read_csv(data_fp, encoding='UTF-8').fillna(value="")
        self.idf = self.init_idf(train_data)

    def init_idf(self, data):
        idf = {}
        q_set = set()
        for index, row in data.iterrows():
            q1 = str(row['spanish_sentence1'])
            q2 = str(row['spanish_sentence2'])
            if q1 not in q_set:
                q_set.add(q1)
                words = q1.lower().split()
                for word in words:
                    idf[word] = idf.get(word, 0) + 1
            if q2 not in q_set:
                q_set.add(q2)
                words = q2.lower().split()
                for word in words:
                    idf[word] = idf.get(word, 0) + 1
        num_docs = len(data)
        for word in idf:
            idf[word] = math.log(num_docs / (idf[word] + 1.)) / math.log(2.)
        return idf

    def extract_row(self, row):
        q1words = {}
        q2words = {}
        for word in str(row['spanish_sentence1']).lower().split():
            if word not in stops:
                q1words[word] = q1words.get(word, 0) + 1
        for word in str(row['spanish_sentence2']).lower().split():
            if word not in stops:
                q2words[word] = q2words.get(word, 0) + 1
        fs = []
        sum_tfidf_q1 = sum([q1words[w] * self.idf.get(w, 0) for w in q1words])
        sum_tfidf_q2 = sum([q2words[w] * self.idf.get(w, 0) for w in q2words])
        if len(q1words) != 0:
            avg_tfidf_q1 = sum([q1words[w] * self.idf.get(w, 0) for w in q1words])/len(q1words)
        else:
            avg_tfidf_q1 = 0

        if len(q2words) != 0:
            avg_tfidf_q2 = sum([q2words[w] * self.idf.get(w, 0) for w in q2words])/len(q2words)
        else:
            avg_tfidf_q2 = 0

        fs.append(sum_tfidf_q1)
        fs.append(sum_tfidf_q2)
        fs.append(avg_tfidf_q1)
        fs.append(avg_tfidf_q2)
        return fs

    def get_feature_num(self):
        return 4