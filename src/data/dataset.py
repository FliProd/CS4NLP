import json

import pandas as pd
from yaml import load
from config import config


class SwissDialDataset():

    def __init__(self, numbers=False):

        self.numbers = numbers        
        self.n_rows = 0

        dataset_name =  config['numbers_dataset'] if numbers else  config['transcribed_dataset']
        self.raw_path = config['raw_data_path'] + dataset_name
        self.processed_path = config['processed_data_path'] + dataset_name


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

            for sentence_set in json_dataset:
                topic = sentence_set.pop('thema')
                del sentence_set['id']

                for key, val in sentence_set.items():
                    rows.append([key, val, topic])
            
            self.n_rows = len(rows)
            print('loaded', self.n_rows, 'row, which represent', self.n_rows/9, 'different sentences')

            df = pd.DataFrame(rows, columns=['dialect', 'sentence', 'topic'])

            df.to_csv(path_or_buf=self.processed_path)
            print('saved dataframe to', self.processed_path)


    def load_processed(self):
        try:
            self.df = pd.read_csv(self.processed_path)
            return True
        except FileNotFoundError:
            return False
