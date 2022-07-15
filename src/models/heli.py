"""
Implementation of the Heli model
"""

from math import log10

import src.utils.utils as util

class Heli():
    def __init__(self, config: dict, dataset: object) -> None:
        self.name = "Heli"
        self.config = config
        self.dataset = dataset
        self.state = {}
        for dialect in self.dataset.config['dialects']:
            self.state[dialect] = {}
            for n in self.config['n']:
                self.state[dialect][f'{n}_grams'] = {}

    def count_ngrams(self, row: dict) -> None:
        dialect = row['dialect']
        if not dialect in self.state.keys():
            return
        for n in self.config['n']:
            nstring = f'{n}_grams'
            for ngram in row[nstring]:
                if self.state[dialect][nstring].get(ngram) == None:
                    self.state[dialect][nstring][ngram] = 1
                else:
                    self.state[dialect][nstring][ngram] += 1

    def get_v_values(self) -> None:
        for dialect in self.dataset.config['dialects']:
            for n in self.config['n']:
                nstring = f'{n}_grams'
                num_ngrams = len(self.state[dialect][nstring].keys())
                for ngram, count in self.state[dialect][nstring].items():
                    self.state[dialect][nstring][ngram] = -log10(count/num_ngrams)

    """
    Main training function of this model
    """
    def train(self) -> None:
        X_train, Y_train = self.dataset.get_train_data()
        X_train.insert(0, 'dialect', Y_train.squeeze())
        df = X_train
        df.apply(lambda x: self.count_ngrams(x), axis=1)
        self.get_v_values()
        return

        

    def get_dg_tn(self, n_grams: list, n: int, dialect:str) -> None:
        nstring = f'{n}_grams'
        dg = 0.0
        for n_gram in n_grams:
            if n_gram in self.state[dialect][nstring]:
                dg += 1.0
        return dg

    def get_vg(self, word: str, n: int, dialect:str) -> None:
        nstring = f'{n}_grams'
        n_grams = util.get_n_grams(word, n)
        dg_tn = self.get_dg_tn(n_grams, n, dialect)
        if dg_tn == 0:
            return self.config["penalty_p"]
        sum = 0.0
        for n_gram in n_grams:
            if n_gram in self.state[dialect][nstring]:
                sum += self.state[dialect][nstring][n_gram]
        return sum / dg_tn

    def predict_dialect(self, row:dict, n:int) -> dict:
        scores = {}
        for dialect in self.dataset.config['dialects']:
            scores[dialect] = 0.0
            for word in row['sentence_version'].split(" "):
                scores[dialect] += self.get_vg(word, n, dialect)
        return (min(scores, key=scores.get))

    """
    Main testing function of this model
    """
    def test(self) -> None:
        X_test, Y_test = self.dataset.get_test_data()
        X_test.insert(0, 'dialect', Y_test.squeeze())
        df = X_test
        predictions = df.apply(lambda x: self.predict_dialect(x, self.config["n_eval"]), axis=1)
        util.evaluate(df, predictions, self.dataset.config["dialects"])

    