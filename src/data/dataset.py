import matplotlib.pyplot as plt

import json

import pandas as pd
import numpy as np
from yaml import load
from config import config


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
        
        path = self.generate_path(kind='processed_dataset')
        if self.load_from_file(path):
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
            self.save_to_file(path)

    """
    Removes the sentences where the dataset does not contain a translation in all languages.
    """
    def balance(self):
        self.properties.append('balanced')

        path = self.generate_path(kind='processed_dataset')
        if not self.load_from_file(path):
            sentence_id_to_count = self.df.groupby('sentence_id')['sentence_version'].count().reset_index(name='count')
            self.df = pd.merge(self.df, sentence_id_to_count, on='sentence_id')
            self.df = self.df[self.df['count'] == 9].drop('count', axis=1)
            self.save_to_file(path)
            print('generated balanced dataframe')


        else:
            print('loaded balanced dataframe')


    """
    Remove special symbols (defined in config) from sentences
    """
    def remove_symbols(self):
        self.properties.append('symRemoved')

        path = self.generate_path(kind='processed_dataset')
        if not self.load_from_file(path):
            self.df['sentence_version'] = self.df['sentence_version'].apply(lambda x: self.remove_from_sentence(x))
            self.save_to_file(path)
            print('generated dataframe without symnbols')
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
    def find_stop_words(self):
        method = config['stopword_calculation_method']
        path = self.generate_path(kind='stopwords')
    
        df_expl_words = self.df.copy()
        # split sentence into list of words
        df_expl_words['sentence_version'] = df_expl_words['sentence_version'].apply(lambda x: x.split(' '))
        # explode list of words to multiple rows
        df_expl_words = df_expl_words.explode('sentence_version').rename(columns={'sentence_version': 'word'})[['dialect', 'word']]

        if method == 'total':
            # count total occurences of each word
            df_tot_occ = df_expl_words.groupby('word').size().sort_values(ascending=False).reset_index(name="count")
            
            # remove words that occur to much
            df_tot_occ = df_tot_occ[df_tot_occ['count'] <= config['stopwords_threshold']['total']]        
            self.stopwords = df_tot_occ['word'].tolist()
            
            self.save_to_file(path)
            print('generated total stopwords')


        elif method == 'tfidf':
            # count total occurences of each word per dialect
            df_tot_occ_dial = df_expl_words.groupby(['word','dialect']).size().sort_values(ascending=False).reset_index(name="total_dial")

            # find max term frequency per dialect
            df_max_occ_dial = df_tot_occ_dial[['dialect', 'total_dial']].groupby('dialect').agg('max').rename(columns={'total_dial': 'max_dial'})

            # merge into single df
            df_tot_max_occ_dial = df_tot_occ_dial.merge(df_max_occ_dial, on='dialect')

            # for each word and dialect calculate its term frequency
            df_tot_max_occ_dial['tf'] = df_tot_max_occ_dial.apply(lambda x: x['total_dial']/x['max_dial'], axis=1)
            df_tf_idf = df_tot_max_occ_dial[['word', 'dialect', 'tf']]

            # calculate idf
            df_num_dial = df_tot_max_occ_dial[['word', 'dialect']].groupby('word').size().reset_index(name='num_dial')
            df_num_dial['idf'] = df_num_dial['num_dial'].apply(lambda x: np.log(9/x))

            df_tf_idf = df_tf_idf.merge(df_num_dial[['word', 'idf']], on='word')

            # merge into single df
            df_tf_idf['tf_idf'] = df_tf_idf['tf'] * df_tf_idf['idf']
            df_tf_idf = df_tf_idf.sort_values(by='tf_idf', ascending=False)

            # use words with a low tf_idf frequency as stopwords
            df_tf_idf.to_csv('test')
            df_tf_idf = df_tf_idf[df_tf_idf['tf_idf'] <=  config['stopwords_threshold']['tfidf']]
            self.stopwords = df_tf_idf['word'].tolist()
            
            self.save_to_file(path)
            print('generated tfidf stopwords')




    """
    Load and store dataset according to preprocessing status
    """
    def load_from_file(self, path):
        try:
            self.df = pd.read_csv(path)
            return True
        except FileNotFoundError:
            return False
    
    def save_to_file(self, path):
        self.df.to_csv(path_or_buf=path, index=False)
        print('saved dataframe to', path)

    def generate_path(self, kind):
        if kind == 'processed_dataset':
            properties = '_'.join(self.properties)
            return self.processed_path + '_' + properties + '.csv'
        elif kind == 'stopwords':
            method = config['stopword_calculation_method']
            return self.processed_path + '_stopwords_' +  method + '_' + str(config['stopwords_threshold'][method])





