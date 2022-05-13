
from tokenize import Expfloat
from config import config
from src.data.dataset_exploration import explore
from src.train.train import train
from src.data.dataset import SwissDialDataset

def main():    

    # load data from json or preprocessed csv
    dataset = SwissDialDataset(numbers=config['numbers'])
    dataset.load_data()
    dataset.balance()
    dataset.remove_symbols()
    dataset.find_stop_words(method='tf_idf')


    #explore(dataset.df)






if __name__ == "__main__":
    main()