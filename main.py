
from tokenize import Expfloat
import pandas as pd

from sklearn.model_selection import train_test_split
from config import config
from src.data.dataset_exploration import explore
from src.train.train import train
from src.data.dataset import SwissDialDataset
from src.models.models import Models

def main():    

    # load data from json or preprocessed csv
    dataset = SwissDialDataset(numbers=config['numbers'])
    dataset.load_data()
    #dataset.balance()
    dataset.remove_symbols()
    #dataset.find_stop_words(method='tf_idf')
    dataset.preprocessing()
    
    #explore(dataset.df)

    models = Models()
    
    df_X = dataset.df.drop(labels=['dialect'], axis=1, inplace=False)
    df_y = dataset.df[['dialect']]
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=420)

    X_train.insert(0, 'dialect', y_train)
    df_train = X_train
    X_test.insert(0, 'dialect', y_test)
    df_test = X_test


    train(df_train, models.models)
    
    






if __name__ == "__main__":
    main()