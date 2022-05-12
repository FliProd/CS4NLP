
from config import config
from data.dataset_exploration import explore
from src.train.train import train
from src.data.dataset import SwissDialDataset

def main():    
    dataset = SwissDialDataset(numbers=config['numbers'])


if __name__ == "__main__":
    main()