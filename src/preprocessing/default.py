"""
Implementation of different preprocessing steps and a wrapper class
"""

import pandas as pd
import os
import numpy as np

import src.utils.utils as util

class DefaultPreprocessor():
    def __init__(self, config: dict) -> None:
        self.config = config
        self.steps_store = [] # List with preprocessing steps that can be stored
        self.steps_no_store = [] # Cannot load n-grams correctly, thus cannot be stored
        self.path = "" # Used to build path where preprocessed data is stored/loaded
        for step in self.config["steps"]:
            if step == "nGrams":
                self.steps_no_store.append(self.create_n_grams)
            elif step == "removeSymbols":
                self.path += step + "_"
                self.steps_store.append(self.remove_symbols)
            elif step == "balance":
                self.path += step + "_"
                self.steps_store.append(self.balance)
            elif step == "removeStopWords":
                self.path += step + "_"
                self.steps_store.append(self.remove_stop_words)
            else:
                print("Invalid preprocessing step. Aborting")
                raise NotImplementedError
        self.path = self.path[:-1]

    def preprocess(self, raw_data: pd.DataFrame, datasetname: str) -> pd.DataFrame:
        # First check if processed data already exists
        path = self.config["processed_data_path"] + datasetname + "/" + self.path + ".csv"
        data = None
        if not self.config["no_load"]:
            data = self.load_processed(path)
        if data is None:
            # If processed data does not exist, do preprocessing
            data = raw_data
            for step in self.steps_store:
                data = step(data, datasetname)
            # Save processed data if congfigured
            if not self.config["no_store"]:
                self.save_processed(path, data)

        # Do preprocessing that cannot be loaded
        for step in self.steps_no_store:
            data = step(data, datasetname)

        return data

    # Load and safe preprocessed data
    def load_processed(self, path) -> pd.DataFrame:
        try:
            data = pd.read_csv(path)
            print("Load data from ", path)
            return data
        except FileNotFoundError:
            return None

    def save_processed(self, path, data: pd.DataFrame) -> None:
        dir_path = path.rsplit('/', 1)[0]
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        data.to_csv(path_or_buf=path, index=False)
        print("Saved data to ", path)


    # Preprocessing steps
    # Every function should have the following structure:
    # def x(self, data: pd.DataFrame, datasetname: str) -> pd.DataFrame:
    # If not, add case destinction in step loop in 'preprocess()'

    """
    Removes the sentences where the dataset does not contain a translation in all languages.
    Note: Only applicable to SwissDial
    """
    def balance(self, data: pd.DataFrame, datasetname: str) -> pd.DataFrame:
        if not datasetname == "SwissDial":
            print("INFO: 'balance' preprocessing step is not applicable to dataset {}. Skipping this step".format(datasetname))
            return data
        sentence_id_to_count = data.groupby('sentence_id')['sentence_version'].count().reset_index(name='count')
        data = pd.merge(data, sentence_id_to_count, on='sentence_id')
        data = data[data['count'] == self.config["n_dialects"] + 1].drop('count', axis=1)
        return data

    """
    Remove special symbols (defined in config) from sentences
    """
    def remove_symbols(self, data: pd.DataFrame, datasetname: str) -> pd.DataFrame:
        data['sentence_version'] = data['sentence_version'].apply(lambda x: util.remove_from_sentence(x, self.config["symbols_to_remove"]))
        return data

    """
    Create n grams for all n defined in config
    """
    def create_n_grams(self, data: pd.DataFrame, datasetname: str, n_s=None) -> pd.DataFrame:
        if n_s is None:
            n_s = self.config["n"]
        data['sentence_version'] = data['sentence_version'].apply(lambda x: x.lower())
        for n in n_s:
            data[f'{n}_grams'] = data['sentence_version'].apply(lambda x: util.get_n_grams_sentence(x, n))
        return data


    """
    Generate array of stopwords by removing words with a frequency over config['stopwords_threshold']
    Then remove those words from dataset
    """
    def remove_stop_words(self, data: pd.DataFrame, datasetname: str, method='total') -> pd.DataFrame:
        df_expl_words = data.copy()
        # split sentence into list of words
        df_expl_words['sentence_version'] = df_expl_words['sentence_version'].apply(lambda x: x.split(' '))
        # explode list of words to multiple rows
        df_expl_words = df_expl_words.explode('sentence_version').rename(columns={'sentence_version': 'word'})[['dialect', 'word']]

        if method == 'total':
            # count total occurences of each word
            df_tot_occ = df_expl_words.groupby('word').size().sort_values(ascending=False).reset_index(name="count")
            # remove words that occur to much
            print(len(df_tot_occ['word']))

            df_tot_occ = df_tot_occ[df_tot_occ['count'] <= self.config['stopwords_threshold_total']]
                        
            print(len(df_tot_occ['word']))

            stopwords = df_tot_occ['word'].tolist()

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
            df_tf_idf = df_tf_idf[df_tf_idf['tf_idf'] <= self.config['stopwords_threshold_tf_idf']]
            stopwords = df_tf_idf['word'].tolist()

        data['sentence_version'] = data['sentence_version'].apply(lambda x: util.remove_from_sentence(x, stopwords))
        return data
