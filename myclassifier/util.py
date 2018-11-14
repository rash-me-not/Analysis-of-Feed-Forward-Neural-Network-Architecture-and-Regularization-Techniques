'''
Created on Oct 21, 2017

@author: mroch
'''

import sys
from keras.models import model_from_json


# Save and Load model functions by Sharath Rao
# https://github.com/sharathrao13/deep-learning/blob/master/utils.py#L57

def save_model(model, name):
    """
    Saves a Keras model to disk as two files: a .json with description of the
    architecture and a .h5 with model weights
    Reference: http://keras.io/faq/#how-can-i-save-a-keras-model
    Parameteres:
    ------------
    model: Keras model that needs to be saved to disk
    name: Name of the model contained in the file names:
        <name>_architecture.json
        <name>_weights.h5
    Returns:
    --------
    True: Completed successfully
    False: Error while saving. The function will print the error.
    """
    try:
        # Uses 'with' to ensure that file is closed properly
        with open(name + '_architecture.json', 'w') as f:
            f.write(model.to_json())
        # Uses overwrite to avoid confirmation prompt
        model.save_weights(name + '_weights.h5', overwrite=True)
        return True  # Save successful
    except:
        print sys.exc_info()  # Prints exceptions
        return False  # Save failed


def load_model(name):
    """
    Loads a Keras model from disk. The model should be contained in two files:
    a .json with description of the architecture and a .h5 with model weights.
    See save_model() to save the model.
    Reference: http://keras.io/faq/#how-can-i-save-a-keras-model
    Parameters:
    -----------
    name: Name of the model contained in the file names:
        <name>_architecture.json
        <name>_weights.h5
    Returns:
    --------
    model: Keras model object.
    """
    # Uses 'with' to ensure that file is closed properly
    with open(name + '_architecture.json') as f:
        model = model_from_json(f.read())
    model.load_weights(name + '_weights.h5')
    return model