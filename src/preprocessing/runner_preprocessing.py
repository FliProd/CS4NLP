"""
This file loads the preprocessing specified in the configuration
"""

from src.preprocessing.default import DefaultPreprocessor
from config import Config

def get_preprocessing(config: Config) -> object:
    preprocessing = config.preprocessing
    if preprocessing["name"] == "default":
        return DefaultPreprocessor(preprocessing)
    else:
        print("Invalid Preprocesing name. Aborting")
        raise NotImplementedError