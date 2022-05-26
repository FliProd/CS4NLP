import matplotlib.pyplot as plt

import json

import pandas as pd
import numpy as np
from yaml import load
from config import config
from nltk import ngrams
import re



class SwissDialDataset():

    def __init__(self, numbers=False):

        self.numbers = numbers        
        self.n_rows = 0

        dataset_name =  config['numbers_dataset'] if numbers else  config['transcribed_dataset']
        self.raw_path = config['raw_data_path'] + dataset_name + '.json'
        self.processed_path = config['processed_data_path'] + dataset_name

        self.properties = []
        self.stopwords = []
        self.stopwords_per_dialect = {}

        self.df = None


    def load_data(self):
        
        if self.load_processed():
            print('loaded dataframe from', self.processed_path)

        else:
            try:
                with open(self.raw_path, 'r') as json_f:
                    json_dataset = json.load(json_f)
            except FileNotFoundError:
                print('raw dataset not found')
                return False
                
            rows = []

            print(len(json_dataset), 'sentences in the dataset')

            for sentence_set in json_dataset:
                id =  sentence_set.pop('id')
                topic = sentence_set.pop('thema')
                code_switching = sentence_set.pop('code_switching', False)

                for key, val in sentence_set.items():
                    rows.append([id, key, val, topic, code_switching])
            
            self.n_rows = len(rows)
            print('loaded', self.n_rows, 'sentence_versions, which represent', self.n_rows/9, 'different sentences')

            self.df = pd.DataFrame(rows, columns=['sentence_id','dialect', 'sentence_version', 'topic', 'code_switching'])
            self.save_processed()

    """
    Removes the sentences where the dataset does not contain a translation in all languages.
    """
    def balance(self):
        self.properties.append('balanced')
        if not self.load_processed():
            sentence_id_to_count = self.df.groupby('sentence_id')['sentence_version'].count().reset_index(name='count')
            self.df = pd.merge(self.df, sentence_id_to_count, on='sentence_id')
            self.df = self.df[self.df['count'] == 9].drop('count', axis=1)
            self.save_processed()
        else:
            print('loaded balanced dataframe')


    """
    Remove special symbols (defined in config) from sentences
    """
    def remove_symbols(self):
        self.properties.append('symRemoved')
        if not self.load_processed():
            self.df['sentence_version'] = self.df['sentence_version'].apply(lambda x: self.remove_from_sentence(x))
            self.save_processed()
        else:
            print('loaded dataframe without symbols')
    
    def remove_from_sentence(self, sentence):
        symbols = config['symbols_to_remove']
        for symbol in symbols:
            sentence = sentence.replace(symbol, '')
        return sentence

    
    """
    Generate array of stopwords by removing words with a frequency over config['stopwords_threshold']
    """
    def find_stop_words(self, method='total'):
        df_expl_words = self.df.copy()
        # split sentence into list of words
        df_expl_words['sentence_version'] = df_expl_words['sentence_version'].apply(lambda x: x.split(' '))
        # explode list of words to multiple rows
        df_expl_words = df_expl_words.explode('sentence_version').rename(columns={'sentence_version': 'word'})[['dialect', 'word']]

        if method == 'total':
            # count total occurences of each word
            df_tot_occ = df_expl_words.groupby('word').size().sort_values(ascending=False).reset_index(name="count")
            # remove words that occur to much
            print(len(df_tot_occ['word']))

            df_tot_occ = df_tot_occ[df_tot_occ['count'] <= config['stopwords_threshold_total']]
                        
            print(len(df_tot_occ['word']))

            self.stopwords = df_tot_occ['word'].tolist()

        elif method == 'tfidf':
            # count total occurences of each word per dialect
            df_tot_occ_dial = df_expl_words.groupby(['word','dialect']).size().sort_values(ascending=False).reset_index(name="total_dial")

            # find max term frequency per dialect
            df_max_occ_dial = df_tot_occ_dial[['dialect', 'total_dial']].groupby('dialect').agg('max').rename(columns={'total_dial': 'max_dial'})

            # merge into single df
            df_tot_max_occ_dial = df_tot_occ_dial.merge(df_max_occ_dial, on='dialect')

            # for each word and dialect calculate its term frequency
            df_tot_max_occ_dial['tf'] = df_tot_max_occ_dial.apply(lambda x: x['total_dial']/x['max_dial'], axis=1)
            print(df_tot_max_occ_dial.head())
            df_tf_idf = df_tot_max_occ_dial[['word', 'dialect', 'tf']]

            # calculate idf
            df_num_dial = df_tot_max_occ_dial[['word', 'dialect']].groupby('word').size().reset_index(name='num_dial')
            df_tf_idf['idf'] = np.log(9 / df_num_dial['num_dial'])

            # merge into single df
            df_tf_idf['tf_idf'] = df_tf_idf['tf'] * df_tf_idf['idf']
            df_tf_idf = df_tf_idf.sort_values(by='tf_idf', ascending=False)

            # use words with a high tf_idf frequency as stopwords
            df_tf_idf = df_tf_idf[df_tf_idf['tf_idf'] <= config['stopwords_threshold_tf_idf']]
            self.stopwords = df_tf_idf['word'].tolist()


    """
    Remove unused data-columns and -rows and create n-grams
    """
    def preprocessing(self):
        self.properties.append('preprocessed')
        # TODO: loading csv results in strings instead of lists for each n_gram list. 
        #if not self.load_processed():
            # remove the high german sentences
        self.df.drop(self.df.index[self.df['dialect'] == 'de'], inplace=True)

        # lowercase the sentences and generate their 4-grams
        self.df['sentence_version'] = self.df['sentence_version'].apply(lambda x: x.lower())
        for n in config['n']:
            self.df[f'{n}_grams'] = self.df['sentence_version'].apply(lambda x: self.create_ngrams(x, n))
            

        # remove the unneeded columns
        self.df.drop(labels=['sentence_id', 'sentence_version', 'topic', 'code_switching'], axis=1, inplace=True)
        self.save_processed()
        #else:
        #    print('loaded preprocessed dataframe')


    def create_ngrams(self, sentence, n):
        n_grams = []
        for i in sentence.split():
            words_n_grams = list(ngrams(' ' + i + ' ', n))
            for word_n_grams in words_n_grams:
                word_n_grams_joined = ''.join(word_n_grams)
                if word_n_grams_joined == " " or word_n_grams_joined == "  ":
                    continue
                n_grams.append(word_n_grams_joined)
        return n_grams



    """
    Load and store dataset according to preprocessing status
    """
    def load_processed(self):
        try:
            self.df = pd.read_csv(self.generate_path())
            return True
        except FileNotFoundError:
            return False
    
    def save_processed(self):
        path = self.generate_path()
        self.df.to_csv(path_or_buf=path, index=False)
        print('saved dataframe to', path)

    def generate_path(self):
        properties = '_'.join(self.properties)
        return self.processed_path + '_' + properties + '.csv'



