"""
This file loads the dataset that is specified in the configuration
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

from src.data.swissdial import SwissDial
from src.data.gdi_vardial import GdiVardial

def get_dataset(dataset_config: dict, preprocessing: object=None) -> object:
    if dataset_config["name"] == "Multiple":
        return MultipleDatasets(dataset_config, preprocessing)
    elif dataset_config["name"] == "SwissDial":
        return SwissDial(dataset_config, preprocessing)
    elif dataset_config["name"].startswith("gdi-vardial"):
        year = dataset_config["name"].split("-")[-1]
        return GdiVardial(dataset_config=dataset_config, year=year, preprocessing=preprocessing)
    else:
        print("Invalid Dataset specified. Aborting")
        raise NotImplementedError

"""
Wrapper class to handle multiple datasets
"""
class MultipleDatasets():
    def __init__(self, config: dict, preprocessing:object) -> None:
        self.config = config
        self.datasets = []
        name=""
        if self.config["no_individual_preprocessing"]:
            for dataset in self.config["datasets"]:
                name += dataset["name"]
                self.datasets.append(get_dataset(dataset))
        else:
            for dataset in self.config["datasets"]:
                self.datasets.append(get_dataset(dataset, preprocessing))

        # Merge datasets
        data = []
        for dataset in self.datasets:
            data.append(dataset.data)
        self.data = pd.concat(data, ignore_index=True)

        if self.config["no_individual_preprocessing"]:
            # Preprocess data if not already done
            self.data = preprocessing.preprocess(raw_data=self.data, datasetname=name)

        self.test_train_split()

    def test_train_split(self) -> None:
        df_X = self.data.drop(labels=['dialect'], axis=1, inplace=False)
        df_Y = self.data[['dialect']]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(df_X, df_Y, test_size=self.config["split"], random_state=42)

    def get_train_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.X_train, self.Y_train

    def get_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.X_test, self.Y_test