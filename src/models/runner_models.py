"""
This file loads the model that is specified in the configuration
"""
from src.models.heli import Heli
from src.models.svm import SVM
from config import Config

def get_model(config: Config, dataset: object) -> object:
    model_name = config.model["name"]
    if model_name == "HeLi":
        return Heli(config=config.model, dataset=dataset)
    elif model_name == "SVM":
        return SVM(config=config.model, dataset=dataset)
    else:
        print("Invalid Model specified. Aborting")
        raise NotImplementedError