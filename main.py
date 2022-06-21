
import argparse

from config import Config
from src.data.runner_dataset import get_dataset
from src.models.runner_models import get_model
from src.preprocessing.runner_preprocessing import get_preprocessing

def run(args):
    config = Config()

    preprocessing = get_preprocessing(config)

    dataset = get_dataset(config.datasets, preprocessing)

    model = get_model(config, dataset)

    if not args.evaluate:
        model.train()

    # TODO: Add step to store model
    
    if not args.train:
        model.test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", action="store_true", help="Only run training")
    parser.add_argument("--evaluate", "-e", action="store_true", help="Only run testing")
    args = parser.parse_args()
    run(args)