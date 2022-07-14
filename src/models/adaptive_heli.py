"""
Implementation of the Heli model
"""

from math import log10
from tokenize import String
from numpy import NaN
import pandas as pd
from typing import Tuple


import src.utils.utils as util

class adaptive_Heli():
    def __init__(self, config: dict, dataset: object) -> None:
        self.config = config
        self.dataset = dataset
        self.state = {}
        for dialect in self.dataset.config['dialects']:
            self.state[dialect] = {}
            for n in self.config['n']:
                self.state[dialect][f'{n}_grams'] = {}
        self.counts = {}
        for dialect in self.dataset.config['dialects']:
            self.counts[dialect] = {}
            for n in self.config['n']:
                self.counts[dialect][f'{n}_grams'] = {}

    def count_ngrams(self, row: dict) -> None:
        dialect = row['dialect']
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
                    self.counts[dialect][nstring][ngram] = count


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

    def get_cm(self, row:dict, n:int):
        scores = {}
        for dialect in self.dataset.config['dialects']:
            scores[dialect] = 0.0
            for word in row['sentence_version'].split(" "):
                scores[dialect] += self.get_vg(word, n, dialect)
        scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}
        count = 0
        
        for index, dialect_score in scores.items():
            if count == 0:
                prediction = index
                r_g = dialect_score
                count += 1
            elif count == 1:
                r_h = dialect_score
                count += 1
            else:
                break
        confidence = r_h - r_g
        return confidence, prediction

    def update_scores(self, row: pd.Series, dialect) -> None:
        for n in self.config['n']:
            nstring = f'{n}_grams'
            for ngram in row[nstring]:
                #print(ngram)
                if self.counts[dialect][nstring].get(ngram) == None:
                    self.counts[dialect][nstring][ngram] = 1
                    self.state[dialect][nstring][ngram] = 0
                else:
                    self.counts[dialect][nstring][ngram] += 1
        for n in self.config['n']:
                nstring = f'{n}_grams'
                num_ngrams = len(self.state[dialect][nstring].keys())
                for ngram in self.state[dialect][nstring].keys():
                    count = self.counts[dialect][nstring][ngram]
                    self.state[dialect][nstring][ngram] = -log10(count/num_ngrams)

    def adaptive_prediction(self, df_test:pd.DataFrame, predictions:pd.Series, cutoff:int, n:int) -> pd.Series: 
        while predictions.isnull().sum() > cutoff:
            if predictions.isnull().sum() % 100 == 0:
                print(predictions.isnull().sum() - cutoff, "NaN values remaining")

            confidence = df_test.apply(lambda x:  pd.Series(self.get_cm(x, self.config["n_eval"]), index=['confidence', 'prediction']), axis=1)
            confidence = confidence.sort_values(by=["confidence"], ascending=False)
            highest_conf = confidence.iloc[0]
            predicted_dialect = highest_conf.loc["prediction"]
            predictions[highest_conf.name] = predicted_dialect
            df_test = df_test.drop(highest_conf.name, axis=0)

        return predictions




    """
    Main testing function of this model
    """
    def test(self) -> None:
        print(f"Running the configuration with \n n_eval={self.config['n_eval']}, \n cutoff={self.config['cutoff']}, \n penalty={self.config['penalty_p']}")

        X_test, Y_test = self.dataset.get_test_data()
        X_test.insert(0, 'dialect', Y_test.squeeze())
        df_test = X_test
        n = self.config["n_eval"]
        X_test = X_test
        predictions = pd.Series(NaN, X_test.index)
        cutoff = int(self.config["cutoff"] * len(predictions.index))
        if cutoff > len(predictions.index) or cutoff < 0: 
            print("Invalid cutoff value, choose between 0 and 1. Aborting")
            raise RuntimeError
        self.adaptive_prediction(X_test, predictions, cutoff, n)
        if cutoff > 0:
            X_test = X_test.drop(X_test[predictions.notna()].index)
            basic_predictions = X_test.apply(lambda x: self.predict_dialect(x, n), axis=1)
            predictions = predictions.combine_first(basic_predictions)
        print(f"Results for the configuration with \n n_eval={self.config['n_eval']}, \n cutoff={self.config['cutoff']}, \n penalty={self.config['penalty_p']}")
        util.evaluate(df_test, predictions, self.dataset.config["dialects"])

    