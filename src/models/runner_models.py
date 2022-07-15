"""
This file loads the model that is specified in the configuration
"""
import pickle
import shutil
import os
from datetime import datetime as dt
from numpy import object0
from src.models.adaptive_heli import adaptive_Heli
from src.models.heli import Heli
from src.models.svm import SVM
from config import Config

def get_model(config: Config, dataset: object) -> object:
    model_name = config.model["name"]
    if model_name == "HeLi":
        return Heli(config=config.model, dataset=dataset)
    elif model_name == "SVM":
        return SVM(config=config.model, dataset=dataset)
    elif model_name == "adaptive_HeLi":
        return adaptive_Heli(config=config.model, dataset=dataset)
    else:
        print("Invalid Model specified. Aborting")
        raise NotImplementedError

def load_model(path: str, dataset: object) -> object:
    try:
        with open('models/' + path + '/model.pickle', 'rb') as f:
            loaded_model = pickle.load(f)
            loaded_model.dataset = dataset
            print("Model stored under {} loaded.".format('models/' + path))
            return loaded_model
    except:
        print("Model could not be loaded. Abort")
        raise NotImplementedError

def store_model(model, name: str) -> None:
    try:
        model_path = model.name + "-" + dt.now().strftime("%Y-%m-%d_%H-%M-%S") if name == "" else name
        # create dir
        dir_path = 'models/' + model_path
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # store model
        with open(dir_path + '/model.pickle', 'wb') as f:
            pickle.dump(model, f)
            print("Model stored under {}.".format(dir_path))
        # store config
        origin = "config.py"
        destination = dir_path + "/config.py"
        shutil.copyfile(origin, destination)
        
    except:
        print("Model could not be stored")
        return