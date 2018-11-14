'''
Created on Sep 30, 2017

@author: mroch
'''

import numpy as np

from keras.callbacks import Callback
from keras.models import Sequential, Model
from keras.utils import np_utils


from sklearn.model_selection import StratifiedKFold



class ErrorHistory(Callback):
    def on_train_begin(self, logs={}):
        self.error = []

    def on_epoch_end(self, batch, logs={}):
        self.error.append(1-logs.get('acc'))


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))



def feed_forward_model(specification):
    """feed_forward_model - specification list
    Create a feed forward model given a specification list
    Each element of the list represents a layer and is formed by a tuple.
    
    (layer_constructor, 
     positional_parameter_list,
     keyword_parameter_dictionary)
    
    Example, create M dimensional input to a 3 layer network with 
    20 unit ReLU hidden layers and N unit softmax output layer
    
    [(Dense, [20], {'activation':'relu', 'input_dim': M}),
     (Dense, [20], {'activation':'relu', 'input_dim':20}),
     (Dense, [N], {'activation':'softmax', 'input_dim':20})
    ]

    """
    model = Sequential()

    for item in specification:
        layertype = item[0]
        # Construct layer and add to model
        # This uses Python's *args and **kwargs constructs
        #
        # In a function call, *args passes each item of a list to 
        # the function as a positional parameter
        #
        # **args passes each item of a dictionary as a keyword argument
        # use the dictionary key as the argument name and the dictionary
        # value as the parameter value
        #
        # Note that *args and **args can be used in function declarations
        # to accept variable length arguments.
        if(len(item) < 3):
            layer = layertype(*item[1])
        else:
            layer = layertype(*item[1], **item[2])
        model.add(layer)

    model.compile(optimizer="Adam",
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    return model


class CrossValidator:
    debug = False

    def __init__(self, Examples, Labels, model_spec, n_folds=10, epochs=10):
        """CrossValidator(Examples, Labels, model_spec, n_folds, epochs)
        Given a list of training examples in Examples and a corresponding
        set of class labels in Labels, train and evaluate a learner
        using cross validation.

        arguments:
        Examples:  feature matrix, each row is a feature vector
        Labels:  Class labels, one per feature vector.  Class labels
            can be strings.
        n_folds:  Number of folds in experiment
        epochs:  Number of times through data set
        model_spec: Specification of model to learn, see
            feed_forward_model() for details and example
        """
        fold = 0;
        kfold = StratifiedKFold(n_folds, shuffle=True)
        self.error = []
        self.losses = []
        self.models = []
        for (train_idx, test_idx) in kfold.split(Examples, Labels):
            fold +=1
            print("Fold #{}".format(fold))
            result = self.train_and_evaluate__model(Examples,
                                                                 Labels,
                                                                 train_idx,
                                                                 test_idx,
                                                                 model_spec)
            self.error.append(result[0])
            self.models.append(result[1])
            self.losses.append(result[2])



    def train_and_evaluate__model(self, examples, labels, train_idx, test_idx,
                                  model_spec, batch_size=10, epochs=10):
        """train_and_evaluate__model(examples, labels, train_idx, test_idx,
                model_spec, batch_size, epochs)
                
        Given:
            examples - List of examples in column major order
                (# of rows is feature dim)
            labels - list of corresponding labels
            train_idx - list of indices of examples and labels to be learned
            test_idx - list of indices of examples and labels of which
                the system should be tested.
            model_spec - Model specification, see feed_forward_model
                for details and example
        Optional arguments
            batch_size - size of minibatch
            epochs - # of epochs to compute
            
        Returns error rate, model, and loss history over training
        """
        loss_history = LossHistory()
        error_history = ErrorHistory()
        onehotlabels = np_utils.to_categorical(labels)
        model = feed_forward_model(model_spec[2])
        model.fit(examples[train_idx], np.asarray(onehotlabels)[train_idx], batch_size, epochs,verbose=0,
                       callbacks=[loss_history,error_history])
        model.evaluate(examples[test_idx], np.asarray(onehotlabels)[test_idx])
        print('ErrorHistory : ',error_history.error)
        print('Loss History : ',loss_history.losses)
        return error_history.error, model, loss_history.losses

    def get_models(self):
        "get_models() - Return list of models created by cross validation"
        return self.models

    def get_errors(self):
        "get_errors - Return list of error rates from each fold"
        return self.errors

    def get_losses(self):
        "get_losses - Return list of loss histories associated with each model"
        return self.losses
