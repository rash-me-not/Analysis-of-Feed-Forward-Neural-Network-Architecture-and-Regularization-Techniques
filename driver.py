'''
Created on Oct 21, 2017

@author: mroch

'''
from mydsp.pca import PCA
from myclassifier.feedforward import CrossValidator, feed_forward_model
import mydsp.features as features
from mydsp.utils import pca_analysis_of_spectra
from mydsp.utils import get_corpus, get_class, Timer
from mydsp.utils import fixed_len_spectrogram, spectrogram, plot_matrix


from keras.layers import Dense, Dropout
from keras import regularizers, metrics
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def main():
    files = get_corpus(
        "C:/Users/rashm/Documents/CS 682/Lab 1/tidigits-isolated-digits-wav/wav/train/woman")
    # for developing
    # if False:
    #     truncate_to_N = 50
    #     print("Truncating t %d files" % (truncate_to_N))
    #     files[truncate_to_N:] = []  # truncate test for speed
    #
    # print("%d files" % (len(files)))

    adv_ms = 10
    len_ms = 20
    # We want to retain offset_s about the center
    offset_s = 0.25

    # -------------------------------------------------------------
    # demo = False
    demo = True
    if demo:
        # Examples showing how to use the different types of spectrograms
        # that can be produced.  Remove this before submitting

        # Standard spectrogram
        (sfull, sfull_t, sfull_f) = spectrogram(files[0], adv_ms, len_ms)
        # Endpoint and truncate
        (strun, strun_t, strun_f) = fixed_len_spectrogram(files[0], adv_ms,
                                                          len_ms, offset_s)
        # Show them
        plt.figure()
        plot_matrix(sfull.T, sfull_t, sfull_f)
        plt.show()
        plt.figure()
        plot_matrix(strun.T, strun_t, strun_f)
        plt.show()


    pca = pca_analysis_of_spectra(files, adv_ms, len_ms, offset_s)
    pca_axis = 0
    for i in range(pca.get_contributions_to_var().shape[0]):
        if (pca.get_contributions_to_var()[i] <= 70):
            pca_axis = pca_axis + 1

    examples = features.extract_features_from_corpus(files, adv_ms, len_ms,
                                                     offset_s, pca=pca,
                                                     pca_axes_N=pca_axis)


    labels = get_class(files)

    model_list = [  [(Dense, [100],
                    {'activation': 'relu', 'input_dim': examples.shape[1],
                     'kernel_regularizer':  regularizers.l1(0.2)}),
                   (Dense, [400], {'activation': 'relu', 'input_dim': 100,
                                  'kernel_regularizer': regularizers.l1(0.2)}),
                   (Dense, [11], {'activation': 'softmax', 'input_dim':100,
                                  'kernel_regularizer': regularizers.l1(0.2)})],

                    [(Dense, [100],
                    {'activation': 'relu', 'input_dim': examples.shape[1],
                     'kernel_regularizer': regularizers.l1_l2(l1=0.01,l2=0.01)}),
                   (Dense, [100], {'activation': 'relu', 'input_dim': 100,
                                  'kernel_regularizer': regularizers.l1_l2(l1=0.01,l2=0.01)}),
                   (Dense, [100], {'activation': 'relu', 'input_dim': 100,
                                   'kernel_regularizer': regularizers.l1_l2(l1=0.01, l2=0.01)}),
                   (Dense, [11], {'activation': 'softmax', 'input_dim':100,
                                  'kernel_regularizer': regularizers.l1_l2(
                                      l1=0.01,l2=0.01)})],

                  [(Dense, [100],
                    {'activation': 'relu', 'input_dim': examples.shape[1],
                     'kernel_regularizer': regularizers.l2(0.005)}),
                   (Dense, [100], {'activation': 'relu', 'input_dim': 100,
                                  'kernel_regularizer': regularizers.l2(0.005)}),
                   (Dense, [100], {'activation': 'relu', 'input_dim': 100,
                                   'kernel_regularizer': regularizers.l2(0.005)}),
                   (Dense, [100], {'activation': 'relu', 'input_dim': 100,
                                   'kernel_regularizer': regularizers.l2(0.005)}),
                   (Dense, [11], {'activation': 'softmax', 'input_dim': 100,
                                   'kernel_regularizer': regularizers.l2(0.005)})],

                    [(Dense, [50],
                      {'activation': 'relu', 'input_dim': examples.shape[1]}),
                     (Dropout, [0.1]),
                     (Dense, [50], {'activation': 'relu', 'input_dim': 50}),
                     (Dropout, [0.1]),
                     (Dense, [50], {'activation': 'relu', 'input_dim': 50}),
                     (Dropout, [0.1]),
                     (Dense, [50], {'activation': 'relu', 'input_dim': 50}),
                     (Dropout, [0.1]),
                     (Dense, [50], {'activation': 'relu', 'input_dim': 50}),
                     (Dropout, [0.1]),
                     (Dense, [11], {'activation': 'softmax', 'input_dim': 50})]]

    N_Folds = 10
    Epochs = 500


    results = CrossValidator(examples, np.asarray(labels), model_list,
                             N_Folds, Epochs)

    print('Training results :\n',results)

    #####################################
    ## Testing phase #####
    #####################################

    test_files = get_corpus(
        "C:/Users/rashm/Documents/CS 682/Lab 1/tidigits-isolated-digits-wav/wav/test/woman")
    models = results.get_models()
    least_error_model_values = np.amin(np.asarray(results.error),axis=1)
    least_error_model_index = np.argmin(least_error_model_values)
    least_error_model = models[least_error_model_index]

    examples_test = features.extract_features_from_corpus(test_files, adv_ms, len_ms,
                                                     offset_s, pca=pca,
                                                     pca_axes_N=pca_axis)


    labels_test = get_class(test_files)
    y_test = np_utils.to_categorical(labels_test)
    y_pred = least_error_model.predict(examples_test)

    y_test_val =np.argmax(y_test,axis=1)
    y_pred_val = np.argmax(y_pred.round(),axis=1)

    cm = confusion_matrix(y_test_val, y_pred_val)
    print('Confusion Matrix : \n', cm)
    correct_labels = 0
    for i in range(examples_test.shape[0]):
        if(y_test_val[i] == y_pred_val[i]):
            correct_labels +=1

    accuracy = correct_labels/examples_test.shape[0] * 100
    print('Accuracy of the best model on Test Data : ', accuracy)


if __name__ == '__main__':
    main()
