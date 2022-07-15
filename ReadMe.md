# Swiss Dialect Identification

This is the code for our project in the computational semantics for NLP course. Developers are Marc Styger, Christopher Raffl and Noah Hampp.

## File Structure
The data/ folder contains the raw data of the GDI VarDial 2027 and 2018 workshops as well as the SwissDial corpus. The src/ contains a pipeline for training and evaluating HeLi, Adaptive HeLi and a SVM on the above datasets. In models/ we store a number of pretrained models. To configure the pipeline use config.py.

## Workflows

To run a certain model on the desired dataset set the following in the config.py file in the root directory.

- choose the model by setting "name" in self.model object (options: HeLi, adaptive_Heli, SVM)
- choose dataset by setting the "name" in self.datasets to one of the names contained in a dataset object in the "datasets" arrays.
- copy the datasets "dialects" array to self.datasets "dialects" field
- set "n_dialects" in self.model to the number of dialects in the chosen dataset
- run the pipeline by running: python main.py in the root directory

To use a preloaded SVM model edit the model_name in src/models/svm.py on line 103 to correspond to one of the names in models/. Similarly set parameters for SVM by setting flags in line 111. Use the comment above to get an overview over the different parameters.

## config.py

- self.model defines which model to use:
    - "name":         Name of the model, matched in /models/runner_models.py
        - Valid options: 
            - "HeLi" -> creates instance of class Heli (/models/heli.py)
            - "SVM"  -> creates instance of class SvM (/models/svm.py)
            - "adaptive_HeLi"   -> creates instance of class adaptive_Heli (/models/adaptive_heli.py)
    - "n":            Size of n-grams used in training. Must be of type list
    - "n_eval":       Size of n-gram used for evaluation. Must be of type int
    - "penalty_p":    Specific value for Heli-model. Penalty added if no n-gram of word present (c.f. eq. 6 in Heli paper)
    - "n_dialects":   number of dialects in the chosen dataset

- self.datasets defines which dataset(s) to use:
    - "name":          Name of the dataset, matched in /data/runner_dataset.py
        - Valid options:
            - "SwissDial":          -> creates instance of class SwissDial (/data/swissdial.py)
            - "gdi-vardial-2017"    -> creates instance of class GdiVardial(/data/gdi_vardial.py)
            - "gdi-vardial-2018"    -> creates instance of class GdiVardial(/data/gdi_vardial.py)
            - "Multiple"            -> Uses wrapper class MultipleDataset(/data/runner_dataset.py) that can handle multiple datasets
    - "split":        Size of test-split. Must be between 0.0 and 1.0
    - "dialects":     dialects that are part of the dataset
    - "datasets":     Only used for MultipleDataset. List of datasets, follow the same structure
    - "datasetnames_to_indices": used to get the correct dataset object
    - "no_individiual_peprocessing":  Only used for MultipleDataset. If set to true, first data is combined and then preprocessed. Otherwise, all datasets are preprocessed individually. Must be of type bool
- self.preprocessing defines which preprocesser to use:
    - "name"          Name of the preprocesser, matched in /preprocessing/runner_preprocessing.py
        Valid options:
            - "default"             -> create instance of class DefaultPreprocessor (/preprocessing/default.py)
    - "steps"         Defines preprocessing steps taken in the given order
        - Valid options:
            - "nGrams"              -> create nGrams ('create_n_grams'). Depends on "n". Should be added as last step
            - "removeSymbols"       -> removes symbols ('remove_symbols'). Depends on "symbols_to_remove".
            - "balance"             -> removes all senteces that are not available in all dialects ('balance'). Only works for SwissDial
            - "removeStopWords"     -> removes stop words ('remove_stop_words'). Depends on "stopwords_threshold_total"
                                            and "stopwords_threshold_tf_idf". UNTESTED
    - "processed_data_path":  Path were processed parts of dataset are stored
    - "no_store":             If set to false, preprocessed data will not be stored. Must be of type bool
    - "n":                    Size of n-grams creates if 'nGrams' is set. Must be of type list
    - "symbols_to_remove":    List of symbols that are removed if 'removeSymbols' is set. Must be of type list
    - "stopwords_threshold_total": 
    - "stopwords_threshold_df_idf":
