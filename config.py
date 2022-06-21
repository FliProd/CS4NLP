class Config():
    def __init__(self):
        self.model = {
            "name": "HeLi",
            "n": [1,2,3,4],
            "n_eval": 4,
            "penalty_p": 5.8
        }
        self.datasets = {
            "name": "Multiple",
            "split": 0.2,
            "dialects": ['ch_sg', 'ch_be', 'ch_gr', 'ch_zh', 'ch_vs', 'ch_bs', 'ch_ag', 'ch_lu'],
            "datasets": [
                {
                    "name": "SwissDial",
                    "raw_data_path": "data/raw/swissdial/sentences_ch_de_transcribed.json",
                    "split": 0.2,
                    "dialects": ['ch_sg', 'ch_be', 'ch_gr', 'ch_zh', 'ch_vs', 'ch_bs', 'ch_ag', 'ch_lu']
                },
                {
                    "name": "gdi-vardial-2017",
                    "raw_data_path": "data/raw/gdi-vardial-2017/combined.txt",
                    "split": 0.2,
                    "dialects": ["ch_bs", "ch_lu", "ch_be", "ch_zh"]
                },
                {
                    "name": "gdi-vardial-2018",
                    "raw_data_path": "data/raw/gdi-vardial-2018/combined.txt",
                    "split": 0.2,
                    "dialects": ["ch_bs", "ch_lu", "ch_be", "ch_zh", "ch_vs"]
                }
            ],
            "no_individual_preprocessing": True
        }
        self.preprocessing = {
            "name": "default",
            "steps": [
                "removeSymbols",
                "nGrams"
            ],
            "processed_data_path": "data/processed/",
            "no_store": False,
            "n": [1,2,3,4],
            "symbols_to_remove": ['"', '«', '»', '.', ',', '?', '!', '(', ')', '[', ']', '&'],
            "stopwords_threshold_total": 200,
            "stopwords_threshold_tf_idf": 0.09
        }

"""
EXPLANATION OF CONFIG:

self.model defines which model to use:
    "name":         Name of the model, matched in /models/runner_models.py
        Valid options: 
            - "HeLi" -> creates instance of class Heli (/models/heli.py)
    "n":            Size of n-grams used in training. Must be of type list
    "n_eval":       Size of n-gram used for evaluation. Must be of type int
    "penalty_p":    Specific value for Heli-model. Penalty added if no n-gram of word present (c.f. eq. 6 in Heli paper)

self.datasets defines which dataset(s) to use:
    "name":          Name of the dataset, matched in /data/runner_dataset.py
        Valid options:
            - "SwissDial":          -> creates instance of class SwissDial (/data/swissdial.py)
            - "gdi-vardial-2017"    -> creates instance of class GdiVardial(/data/gdi_vardial.py)
            - "gdi-vardial-2018"    -> creates instance of class GdiVardial(/data/gdi_vardial.py)
            - "Multiple"            -> Uses wrapper class MultipleDataset(/data/runner_dataset.py) that can handle multiple datasets
    "split":        Size of test-split. Must be between 0.0 and 1.0
    "dialects":     dialects that are part of the dataset
    "datasets":     Only used for MultipleDataset. List of datasets, follow the same structure
    "no_individiual_peprocessing":  Only used for MultipleDataset. If set to true, first data is combined and then preprocessed.
                                        Otherwise, all datasets are preprocessed individually. Must be of type bool
self.preprocessing defines which preprocesser to use:
    "name"          Name of the preprocesser, matched in /preprocessing/runner_preprocessing.py
        Valid options:
            - "default"             -> create instance of class DefaultPreprocessor (/preprocessing/default.py)
    "steps"         Defines preprocessing steps taken in the given order
        Valid options:
            - "nGrams"              -> create nGrams ('create_n_grams'). Depends on "n". Should be added as last step
            - "removeSymbols"       -> removes symbols ('remove_symbols'). Depends on "symbols_to_remove".
            - "balance"             -> removes all senteces that are not available in all dialects ('balance'). Only works for SwissDial
            - "removeStopWords"     -> removes stop words ('remove_stop_words'). Depends on "stopwords_threshold_total"
                                            and "stopwords_threshold_tf_idf". UNTESTED
    "processed_data_path":  Path were processed parts of dataset are stored
    "no_store":             If set to false, preprocessed data will not be stored. Must be of type bool
    "n":                    Size of n-grams creates if 'nGrams' is set. Must be of type list
    "symbols_to_remove":    List of symbols that are removed if 'removeSymbols' is set. Must be of type list
    "stopwords_threshold_total": 
    "stopwords_threshold_df_idf":
"""