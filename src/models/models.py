from config import config



class Models:

    def __init__(self):
        self.models = {}
        for dialect in config['dialects']:
            self.models[dialect] = {}
            for n in config['n']:
                self.models[dialect][f'{n}_grams'] = {}


        