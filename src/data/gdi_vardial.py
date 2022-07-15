"""
Implementation of the gdi-vardial datasets
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

class GdiVardial():
    # maps dialects used in gdi-vardial to SwissDial format
    dialect_mapping = {
        "LU": "ch_lu",
        "BE": "ch_be",
        "ZH": "ch_zh",
        "BS": "ch_bs",
        "XY": "ch_vs",
    }
    def __init__(self, dataset_config:dict, year: str, preprocessing:object) -> None:
        self.config = dataset_config
        self.year = year
        self.preprocessing = preprocessing

        self.load_data()
        if preprocessing is not None:
            self.data = self.preprocessing.preprocess(raw_data=self.data, datasetname=self.config["name"])
        self.test_train_split()

    def load_data(self) -> None:
        try:
            rows = []
            with open(self.config["raw_data_path"], 'r') as f:
                lines = f.readlines()
            for line in lines:
                splitted_line = line.rsplit("\t", 1)
                sentence = splitted_line[0]
                dialect = splitted_line[1].replace("\n", "")
                dialect = GdiVardial.dialect_mapping[splitted_line[1].replace("\n", "")]
                if not dialect in self.config["dialects"]:
                    continue
                rows.append([dialect, sentence])
            self.n_rows = len(rows)
            self.data = pd.DataFrame(rows, columns=['dialect', 'sentence_version'])
        except:
            print("Dataset not found, invalid path. Abort")
            raise NotImplementedError

    def test_train_split(self) -> None:
        df_X = self.data.drop(labels=['dialect'], axis=1, inplace=False)
        df_Y = self.data[['dialect']]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(df_X, df_Y, test_size=self.config["split"], random_state=42)

    def get_train_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.X_train, self.Y_train

    def get_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.X_test, self.Y_test