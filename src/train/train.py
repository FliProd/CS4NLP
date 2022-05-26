from math import log10
from config import config

def count_ngrams(row, models):
    dialect = row['dialect']
    for n in config['n']:
        nstring = f'{n}_grams'
        for ngram in row[nstring]:
            if models[dialect][nstring].get(ngram) == None:
                models[dialect][nstring][ngram] = 1
            else:
                models[dialect][nstring][ngram] += 1


def get_v_values(models):
    for dialect in config['dialects']:
        for n in config['n']:
            nstring = f'{n}_grams'
            num_ngrams = len(models[dialect][nstring].keys())
            for ngram, count in models[dialect][nstring].items():
                models[dialect][nstring][ngram] = -log10(count/num_ngrams)
            

def train(df, models):
    df.apply(lambda x: count_ngrams(x, models), axis=1)
    get_v_values(models)
   