# Project Title

This is the code for our project in the computational semantics for NLP course. Developers are Marc Styger, Christopher Raffl and Noah Hampp.

## File Structure
The data/ folder contains the raw data of the GDI VarDial 2027 and 2018 workshops as well as the SwissDial corpus. The src/ contains a pipeline for training and evaluating HeLi, Adaptive HeLi and a SVM on the above datasets. In models/ we store a number of pretrained models. To configure the pipeline use config.py.

## Workflows

To run a certain model on the desired dataset set the following in the config.py file in the root directory.

- choose the model by setting "name" in self.model object (options: HeLi, SVM)
- choose dataset by setting the "name" in self.datasets to one of the names contained in a dataset object in the "datasets" arrays.
- copy the datasets "dialects" array to self.datasets "dialects" field
- set "n_dialects" in self.model to the number of dialects in the chosen dataset
- run the pipeline by running: python main.py in the root directory

To use a preloaded SVM model edit the model_name in src/models/svm.py on line 103 to correspond to one of the names in models/. Similarly set parameters for SVM by setting flags in line 111. Use the comment above to get an overview over the different parameters.


