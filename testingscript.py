import argparse
import copy

from numpy import test

from main import run
from config import Config

def main(args):
    config = Config()
    datasetcombinations = [
        {
            "name": "SwissDial",
            "datasets": ["SwissDial"]
        },
        {
            "name": "VarDial-17",
            "datasets": ["VarDial-17"]
        },
        {
            "name": "VarDial-18",
            "datasets": ["VarDial-18"]
        },
        {
            "name": "SwissDial-VarDial-17",
            "datasets": ["SwissDial", "VarDial-17"]
        },
        {
            "name": "SwissDial-VarDial-18",
            "datasets": ["SwissDial", "VarDial-18"]
        },
        {
            "name": "VarDial-17-VarDial-18",
            "datasets": ["VarDial-17", "VarDial-18"]
        },
        {
            "name": "SwissDial-VarDial-17-VarDial-18",
            "datasets": ["SwissDial", "VarDial-17", "VarDial-18"]
        },
    ]
    dataset_config = {
        "SwissDial": {
            "name": "SwissDial",
            "raw_data_path": "data/raw/swissdial/sentences_ch_de_transcribed.json",
            "split": 0.2,
            "dialects": ['ch_sg', 'ch_be', 'ch_gr', 'ch_zh', 'ch_vs', 'ch_bs', 'ch_ag', 'ch_lu'],
        },
        "VarDial-17": {
            "name": "gdi-vardial-2017",
            "raw_data_path": "data/raw/gdi-vardial-2017/combined.txt",
            "split": 0.2,
            "dialects": ["ch_bs", "ch_lu", "ch_be", "ch_zh"]
        },
        "VarDial-18": {
            "name": "gdi-vardial-2018",
            "raw_data_path": "data/raw/gdi-vardial-2018/combined.txt",
            "split": 0.2,
            "dialects": ["ch_bs", "ch_lu", "ch_be", "ch_zh", "ch_vs"]
        },
        "multiple": {
            "name": "Multiple",
            "split": 0.2,
            "dialects": [],
            "datasets": [],
            "no_individual_preprocessing": True
        }
    }
    if not args.testing:
        args.train = True
        args.store = True
        args.evaluate = False
        for combination in datasetcombinations:
            if len(combination["datasets"]) == 1:
                config.datasets = copy.deepcopy(dataset_config[combination["datasets"][0]])
                config.datasets["dialects"].sort()
            else:
                config.datasets = copy.deepcopy(dataset_config["multiple"])
                for dataset in combination["datasets"]:
                    config.datasets["datasets"].append(copy.deepcopy(dataset_config[dataset]))
                    config.datasets["dialects"] = list(set(config.datasets["dialects"]) | set(dataset_config[dataset]["dialects"]))
                config.datasets["dialects"].sort()
                for dataset in config.datasets["datasets"]:
                    dataset["dialects"] = config.datasets["dialects"]
            args.name = combination["name"]
            run(args, config)

    if not args.training:
        args.train = False
        args.store = False
        args.evaluate = True
        for train_combination in datasetcombinations:
            for test_combination in datasetcombinations:
                if len(test_combination["datasets"]) == 1:
                    config.datasets = copy.deepcopy(dataset_config[test_combination["datasets"][0]])
                    config.datasets["dialects"].sort()
                else:
                    config.datasets = copy.deepcopy(dataset_config["multiple"])
                    for dataset in test_combination["datasets"]:
                        config.datasets["datasets"].append(copy.deepcopy(dataset_config[dataset]))
                        config.datasets["dialects"] = list(set(config.datasets["dialects"]) | set(dataset_config[dataset]["dialects"]))
                train_dialects = []
                if len(train_combination["datasets"]) == 1:
                    train_dialects = dataset_config[train_combination["datasets"][0]]["dialects"]
                else:
                    for dataset in train_combination["datasets"]:
                        train_dialects = list(set(train_dialects) | set(dataset_config[dataset]["dialects"]))
                config.datasets["dialects"] = list(set(train_dialects) & set(config.datasets["dialects"]))
                config.datasets["dialects"].sort()
                if len(test_combination["datasets"]) > 1:
                    for dataset in config.datasets["datasets"]:
                        dataset["dialects"] = config.datasets["dialects"]
                args.name = train_combination["name"]
                print("Testing {} on model trained on {}".format(test_combination["name"], train_combination["name"]))
                run(args, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", "-t", action="store_true", help="Only run training")
    parser.add_argument("--testing", "-e", action="store_true", help="Only run testing")
    args = parser.parse_args()
    main(args)
