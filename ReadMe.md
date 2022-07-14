# Project Title

run:
choose daataset with name
set the right dialects
set n_diaalects in config for svm

loaad svm model:
change model_name in line 103 in src/models/svm.py

change svm parameters:
change parameters in line 111 aaccording to below documentation
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
## Project Description
ToDo

## Dataset
ToDo

## Environment
ToDo

