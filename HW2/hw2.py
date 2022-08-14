import ssl
from collections import defaultdict
from dataclasses import dataclass
from os.path import join as pj
from typing import Tuple, List

import conllutils
import numpy as np
import pandas as pd
from nltk.tag import tnt

try:
    # noinspection PyUnresolvedReferences,PyProtectedMember
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# from time import time
# import matplotlib.pyplot as plt
# import pandas as pd


def get_dataset(file_path: str) -> List[List[Tuple[str, str]]]:
    data = []
    for s in conllutils.read_conllu(file=file_path):
        data.append([(s[i]['form'], s[i]['xpos']) for i in range(len(s)) if not isinstance(s[i]['id'], tuple)])

    return data


_train_data = get_dataset(file_path=pj('en_gum-ud-train.conllu'))
_dev_data = get_dataset(file_path=pj('en_gum-ud-dev.conllu'))
_test_data = get_dataset(file_path=pj('en_gum-ud-test.conllu'))


@dataclass(init=True)
class Results:
    words_accuracy: float
    sentences_accuracy: float


class SimpleTagger:
    def train(self, data: List[List[Tuple[str, str]]]):
        word_map = defaultdict(lambda: defaultdict(lambda: 0))
        tag_map = defaultdict(lambda: 0)

        for sentence in data:
            for word, tag in sentence:
                word_map[word][tag] += 1
                tag_map[tag] += 1

        self.max_tag = max(tag_map, key=tag_map.get)

        self.map = {}
        for word in word_map:
            self.map[word] = max(word_map[word], key=word_map[word].get)

    def evaluate(self, data: List[List[Tuple[str, str]]]) -> Results:
        total_words, words_correctly_predicted, sentence_correctly_predicted = 0, 0, 0

        for sentence in data:
            word_successe_in_sentence = sum([self.map.get(word, self.max_tag) == tag for word, tag in sentence])

            total_words += len(sentence)
            words_correctly_predicted += word_successe_in_sentence
            sentence_correctly_predicted += (len(sentence) == word_successe_in_sentence)

        return Results(words_accuracy=words_correctly_predicted / total_words,
                       sentences_accuracy=sentence_correctly_predicted / len(data))


_simple_tagger = SimpleTagger()
_simple_tagger.train(data=_train_data)
_simple_dev_acc = _simple_tagger.evaluate(data=_dev_data)
_simple_test_acc = _simple_tagger.evaluate(data=_test_data)


class HMM_Tagger:
    def train(self, data: List[List[Tuple[str, str]]]):
        # word & tag mapping
        word_map, tag_map = {}, {}

        for sentence in data:
            for word, tag in sentence:
                if word not in word_map:
                    word_map[word] = {'id': len(word_map), 'tags': defaultdict(lambda: 0)}
                word_map[word]['tags'][tag] += 1

                if tag not in tag_map:
                    tag_map[tag] = {'id': len(tag_map), 'count': 1}
                else:
                    tag_map[tag]['count'] += 1

        # mapping id to word and word to id
        id_to_word = {word_map[word]['id']: word for word in word_map.keys()}
        word_to_id = {v: k for k, v in id_to_word.items()}

        # mapping id to tag and tag to id
        id_to_tag = {tag_map[tag]['id']: tag for tag in tag_map}
        tag_to_id = {v: k for k, v in id_to_tag.items()}

        # create Pi
        tag_map_values = np.array([tag['count'] for tag in tag_map.values()])
        # noinspection PyPep8Naming
        Pi = tag_map_values / tag_map_values.sum()

        # create B
        # noinspection PyPep8Naming
        B = np.zeros((len(tag_map), len(word_map)))
        for word in word_map.values():
            for tag in word['tags']:
                B[tag_map[tag]['id'], word['id']] = word['tags'][tag]
        B /= B.sum(axis=1)[:, np.newaxis]

        # create A
        # noinspection PyPep8Naming
        A = np.zeros((len(tag_map), len(tag_map)))
        for sentence in data:
            for i in range(len(sentence) - 1):
                A[tag_map[sentence[i][1]]['id'], tag_map[sentence[i + 1][1]]['id']] += 1
        A /= A.sum(axis=1)[:, np.newaxis]

        self.word_to_id = word_to_id
        self.tag_to_id = tag_to_id
        self.tag_id = list(id_to_tag.keys())
        self.A = A
        self.B = B
        self.Pi = Pi

    def evaluate(self, data: List[List[Tuple[str, str]]]) -> Results:
        total_words, words_correctly_predicted, sentence_correctly_predicted = 0, 0, 0

        for sentence in data:
            n_sen = len(sentence)
            tags = np.zeros(n_sen)
            words = np.zeros(n_sen, dtype=int)

            for i, (word, tag) in enumerate(sentence):
                words[i] = self.word_to_id.get(word, -1)
                tags[i] = self.tag_to_id.get(tag, -1)

            split_points = np.where(words == -1)[0]
            if 0 not in split_points:
                split_points = np.append([0], split_points)
            if n_sen not in split_points:
                split_points = np.append(split_points, [n_sen])

            # generate pairs
            sequences = [words[split_points[i]:split_points[i + 1]]
                         for i in range(len(split_points) - 1)]
            result = np.array([])

            for sequence in sequences:
                if sequence[0] == -1:  # if -1 in the sequence
                    result = np.append(result, np.random.choice(self.tag_id, size=1))
                    if len(sequence) > 1:
                        result = np.append(result, viterbi(observations=sequence[1:], A=self.A, B=self.B, Pi=self.Pi))
                else:
                    result = np.append(result, viterbi(observations=sequence, A=self.A, B=self.B, Pi=self.Pi))

            total_words += n_sen
            words_correctly_predicted += sum(result == tags)
            sentence_correctly_predicted += (sum(result == tags) == n_sen)

        return Results(words_accuracy=words_correctly_predicted / total_words,
                       sentences_accuracy=sentence_correctly_predicted / len(data))


def viterbi(observations: List[int], A: np.ndarray, B: np.ndarray, Pi: np.ndarray) -> np.ndarray:
    n_obs = len(observations)
    n_tags = A.shape[0]

    delta = np.zeros((n_tags, n_obs))
    psi = np.zeros((n_tags, n_obs))

    delta[:, 0] = B[:, observations[0]] * Pi

    for i in range(1, n_obs):
        for j in range(0, n_tags):
            trans_p = delta[:, i - 1] * A[:, j]
            max_idx = np.argmax(trans_p)
            max_value = trans_p[max_idx]
            psi[j, i], delta[j, i] = max_idx, max_value
            delta[j, i] = delta[j, i] * B[j, observations[i]]

    seq = np.zeros(n_obs, dtype=int)
    seq[n_obs - 1] = np.argmax(delta[:, n_obs - 1])
    for i in range(n_obs - 1, 0, -1):
        seq[i - 1] = psi[seq[i], i]

    return seq


_hmm_tagger = HMM_Tagger()
_hmm_tagger.train(data=_train_data)
_hmm_dev_acc = _hmm_tagger.evaluate(data=_dev_data)
_hmm_test_acc = _hmm_tagger.evaluate(data=_test_data)


# tnt_pos_tagger = tnt.TnT()
# tnt_pos_tagger.train(_train_data)
# print(tnt_pos_tagger.accuracy(_test_data))  # 0.8357178095707942


class TnT_Tagger:
    def __init__(self):
        self.tagger = tnt.TnT()

    def train(self, data: List[List[Tuple[str, str]]]):
        self.tagger.train(data=data)

    def evaluate(self, data: List[List[Tuple[str, str]]]) -> Results:
        sentence_correctly_predicted = 0

        for sentence in data:
            words = list(map(lambda tp: tp[0], sentence))
            sentence_correctly_predicted += (self.tagger.tag(words) == sentence)

        # Note - word accuracy can be calculated also in the loop to save time if needed.
        return Results(words_accuracy=self.tagger.evaluate(gold=data),
                       sentences_accuracy=sentence_correctly_predicted / len(data))


_tnt_tagger = TnT_Tagger()
_tnt_tagger.train(data=_train_data)
_memm_dev_acc = _tnt_tagger.evaluate(data=_dev_data)
_memm_test_acc = _tnt_tagger.evaluate(data=_test_data)

_dw_acc = [_simple_dev_acc.words_accuracy, _hmm_dev_acc.words_accuracy, _memm_dev_acc.words_accuracy]
_ds_acc = [_simple_dev_acc.sentences_accuracy, _hmm_dev_acc.sentences_accuracy, _memm_dev_acc.sentences_accuracy]

_tw_acc = [_simple_test_acc.words_accuracy, _hmm_test_acc.words_accuracy, _memm_test_acc.words_accuracy]
_ts_acc = [_simple_test_acc.sentences_accuracy, _hmm_test_acc.sentences_accuracy, _memm_test_acc.sentences_accuracy]

_results = {
    'Dev Word Accuracy': _dw_acc,
    'Dev Sentence Accuracy': _ds_acc,
    'Test Word Accuracy': _tw_acc,
    'Test Sentence Accuracy': _ts_acc
}

_cols = ['Dev Word Accuracy', 'Dev Sentence Accuracy', 'Test Word Accuracy', 'Test Sentence Accuracy']
pd.DataFrame(data=_results, columns=_cols, index=['Simple Tagger', 'HMM Tagger', 'MEMM Tagger'])
