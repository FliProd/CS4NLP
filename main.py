
import argparse

from config import Config
from src.data.runner_dataset import get_dataset
from src.models.runner_models import get_model, store_model, load_model
from src.preprocessing.runner_preprocessing import get_preprocessing

def run(args, config=None):
    config = Config() if config == None else config

    preprocessing = get_preprocessing(config)

    dataset = get_dataset(config.datasets, preprocessing)

    if not args.evaluate:
        model = get_model(config, dataset)
        model.train()
        if args.store:
            store_model(model, args.name)
    
    if not args.train:
        if args.name:
            model = load_model(args.name, dataset)
        model.test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", action="store_true", help="Only run training")
    parser.add_argument("--store", "-s", action="store_true", help="Store trained model")
    parser.add_argument("--evaluate", "-e", action="store_true", help="Only run testing")
    parser.add_argument("--name", "-n", type=str, help="Name of the model to load/store (i.e. the directory)")
    args = parser.parse_args()
    run(args)