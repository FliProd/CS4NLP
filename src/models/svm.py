"""
Implementation of the SVM model
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import scipy
from libsvm.svmutil import *


import src.utils.utils as util

class SVM():
    def __init__(self, config: dict, dataset: object) -> None:
        self.config = config
        self.dataset = dataset

        self.feature_encoders = []
        self.target_encoder = MultiLabelBinarizer()

        self.models = []
        self.n = config['n_dialects']
  

    def init_feature_encoders(self):
        X_train, _ = self.dataset.get_train_data()
        X_test, _ = self.dataset.get_test_data()

        for n in self.config['n']:
            n_grams_per_sentence = X_train[f'{n}_grams'].tolist() + X_test[f'{n}_grams'].tolist()

            encoder = MultiLabelBinarizer()
            encoder.fit(n_grams_per_sentence)

            self.feature_encoders.append(encoder)

    def init_target_encoder(self):
        _, Y_train = self.dataset.get_train_data()
        _, Y_test = self.dataset.get_test_data()
    
        self.target_encoder.fit_transform(np.array(Y_train['dialect'].tolist() + Y_test['dialect'].tolist()).reshape(-1, 1))

    def encode_features(self, X, predict):
        # use the multilabelbinarizer for all n we have n-grams for and concatenate to single feature matrix
        X_encoded_split = []
        for n in self.config['n']:
            if predict:
                X_encoded_part = self.feature_encoders[n-1].transform([X[f'{n}_grams']])
            else:
                X_encoded_part = self.feature_encoders[n-1].transform(X[f'{n}_grams'].tolist())
            X_encoded_split.append(X_encoded_part)

        X_encoded = np.concatenate(X_encoded_split, axis=1)

        return X_encoded


    """
    Main training function of this model
    """
    def train(self) -> None:
        X_train, Y_train = self.dataset.get_train_data()

        self.init_feature_encoders()
        self.init_target_encoder()

        X_encoded = self.encode_features(X_train, predict=False)

        Y_encoded = self.target_encoder.transform(np.array(Y_train['dialect'].tolist()).reshape(-1, 1))

        X_encoded_wrapped = scipy.asarray(X_encoded)
        Y_encoded_wrapped = scipy.asarray(Y_encoded)

        """
        Parameters
        -s svm_type : set type of SVM (default 0)
            0 -- C-SVC
            1 -- nu-SVC
            2 -- one-class SVM
            3 -- epsilon-SVR
            4 -- nu-SVR
        -t kernel_type : set type of kernel function (default 2)
            0 -- linear: u'*v
            1 -- polynomial: (gamma*u'*v + coef0)^degree
            2 -- radial basis function: exp(-gamma*|u-v|^2)
            3 -- sigmoid: tanh(gamma*u'*v + coef0)
        -d degree : set degree in kernel function (default 3)
        -g gamma : set gamma in kernel function (default 1/num_features)
        -r coef0 : set coef0 in kernel function (default 0)
        -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
        -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
        -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
        -m cachesize : set cache memory size in MB (default 100)
        -e epsilon : set tolerance of termination criterion (default 0.001)
        -h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
        -b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
        -wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)
        -q output: suppress output to terminal
        """

        # train a  model per dialect
        for i in range(0,self.n):
            model_name = 'models/d' + str(i) + '_vardial17_linear_c1.model'
            # try and load model from storage
            m = svm_load_model(model_name)

            # train model if none could be loaded
            if m is None:
                print('training model', i)
                prob  = svm_problem(Y_encoded_wrapped[:,i], X_encoded_wrapped)
                param = svm_parameter('-t 0 -c 1 -b 1 -h 0')  
                m = svm_train(prob, param)
                svm_save_model(model_name, m)
            else:
                print('loaded model', i)
            
            self.models.append(m)




    def predict_dialect(self, row:dict, n:int) -> dict:
        # encode x
        x_encoded = self.encode_features(row, predict=True)
        
        # let each dialect svm predict a probability
        probs = np.zeros((8))
        for i in range(0,self.n):
            # p_label seems to not really work, p_acc is not informative since we dont have verification here
            p_label, p_acc, p_val = svm_predict([0], x_encoded, self.models[i], '-b 1 -q')
            
            # need to know in what order the svm encountered the 0/1 labels standiung for not/in dialect
            # p_val contains class probabilities: [[P(x in dialect i), P(x not in dialect i)]]
            first_label_encountered = self.models[i].label[0]
            if first_label_encountered == 0:
                probs[i] = p_val[0][1]
            else:
                probs[i] = p_val[0][0]

        
        # predict class as the one with the highest confidence
        p_label_index = np.argmax(probs)
        # give back class as string
        p_label = self.target_encoder.classes_[p_label_index]


        return p_label


    """
    Main testing function of this model
    """
    def test(self) -> None:
        X_test, Y_test = self.dataset.get_test_data()
        X_test.insert(0, 'dialect', Y_test.squeeze())
        df = X_test
        predictions = df.apply(lambda x: self.predict_dialect(x, self.config["n_eval"]), axis=1)
        util.evaluate(df, predictions, self.dataset.config["dialects"])

    