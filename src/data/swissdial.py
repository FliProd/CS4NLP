"""
Implementation of the Swiss Dial dataset
"""

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

class SwissDial():
    def __init__(self, dataset_config: dict, preprocessing: object) -> None:
        self.config = dataset_config
        self.preprocessing = preprocessing

        self.load_data()
        if preprocessing is not None:
            self.data = self.preprocessing.preprocess(raw_data=self.data, datasetname=self.config["name"])
        self.remove_unused()
        self.test_train_split()

    def remove_unused(self) -> None:
        self.data.drop(self.data.index[~self.data['dialect'].isin(self.config["dialects"])], inplace=True)
        self.data.drop(labels=['sentence_id', 'topic', 'code_switching'], axis=1, inplace=True)

    def load_data(self) -> None:
        try:
            with open(self.config["raw_data_path"], 'r') as json_f:
                json_dataset = json.load(json_f)
        except FileNotFoundError:
            print("Dataset not found, invalid path. Abort")
            raise NotImplementedError
        rows = []
        for sentence_set in json_dataset:
            id = sentence_set.pop('id')
            topic = sentence_set.pop('thema')
            code_switching = sentence_set.pop('code_switching', False)

            for key, val in sentence_set.items():
                rows.append([id, key, val, topic, code_switching])
        
        self.n_rows = len(rows)
        
        self.data = pd.DataFrame(rows, columns=['sentence_id', 'dialect', 'sentence_version', 'topic', 'code_switching'])

    def test_train_split(self) -> None:
        df_X = self.data.drop(labels=['dialect'], axis=1, inplace=False)
        df_Y = self.data[['dialect']]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(df_X, df_Y, test_size=self.config["split"], random_state=42)

    def get_train_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.X_train, self.Y_train

    def get_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.X_test, self.Y_test