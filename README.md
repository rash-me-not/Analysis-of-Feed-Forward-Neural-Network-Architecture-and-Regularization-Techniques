# Analysis-of-Feed-Forward-Neural-Network-Architecture-and-Regularization-Techniques
Analysis of Feed-Forward Neural Network Architecture and Regularization Techniques

Abstract
In this paper, we are using the MIT/TIDIGITS corpus (Leonard and Doddington, 1993) to explore the performance and accuracy of different types of Feed Forward Neural Network Architectures. The different approaches currently utilized in the experiment include Regularization techniques like L1, L2 and Dropout, Dimensionality reduction techniques like PCA, and Deep vs Wide Neural Network architecture with different activation functions. 

Introduction
A common problem that arises while training huge neural network architectures is Overfitting, where the model tries to closely learn the extreme details in the training set and eventually ends up giving a huge generalization error. In this lab we are going to regularize the models in order to limit their capacity by adding parameter norm penalties or by using the Dropout technique. The parameter normalization techniques used in this paper are L1 and L2 Regularization which are discussed in the below section.
L2 Regularization:
The L2 parameter norm penalty, also known as weight decay drives w closer to the origin by adding the regularization term.
Ω(w)  =  1/2 〖||w||〗_2^2  =  1/2 w^Tw
The update rule of gradient decent using L2 norm penalty is w ← (1 − ∈α)w − ∈∇w J(w)
Here, the weights multiplicatively shrink by a constant factor at each step before performing the gradient update, thereby reducing the overfitting problem by selecting the appropriate value of penalizing term
L1 Regularization:
The L1 Regularization provides solution that have a sparse representation, where the sparsity corresponds to a type of feature selection mechanism. Here the L1 penalty can be represented as :
Ω(θ) = ||w|| = ∑_i▒〖|w_i  |〗
In this experiment, we are going to implement the L1 and L2 regularization with help of the kernel_regularizer instance in  keras.regularizers.Regularizer library
In order to regularize a fixed size model, we also use the dropout model while training the model. It involves temporarily dropping out random hidden and visible nodes from the neural network along with all its input and output connections, and averaging out the predictions of all its possible settings. In this lab, we experiment the accuracy of the network using Dropout with different sets of probability at different layers of the sequential model.
Since the MIT/TIDIGITS corpus dataset is large, introducing the PCA can help us retain the original variables into a smaller set of new dimensions, with a minimum loss of information. With PCA the rate of convergence for the Loss function is quicker by retrieving the important features of a large data set. 

Experiment Setup:
The Feed Forward Neural network is implemented on Python 3.6 using TensorFlow 1.11 and the Keras library 2.2.4. In order to proceed with the model prediction, we are using the speech endpoint detection using the Gaussian Mixture Model to separate the input corpus data from the noisy environment. The below spectrograms can be used to visualize the speech end pointer for the first audio file in the TIDIGITS corpus dataset. Figure 1 represents the spectrum of frequencies varying over a time duration of 0.7sec for the audio signal in first file ‘1a.wav’  . Figure 2 represents a fixed length spectrogram of the same signal obtained by offsetting the centre of the speech signal with a time duration of 0.25sec in order to endpoint the speech data.
  



The fixed length spectrogram is used to extract the features from the corpus data set.  A 2d array of features space is used to train the model where each row of the feature space corresponds to a feature vector of the filenames passed in files. The model is trained to predict the class labels of the files where each class label denotes the digit in the filename (For example: the first two feature vectors extracted from the corpus dataset will belong to class label of 1 since they have the digits 1 in their filename – file 1a.wav , 1b.wav). In order to train the model we are using the ‘Stratified K-Folds’ cross-validator from the scikit-learn library , where the the original sample is randomly partitioned into k equal sized subsets. Of the k subset, a single set is retained for validating the model, and the remaining k − 1 sets are used as training data.

Experiments:
	Using L2 Regularization on Deep vs Wide Network with PCA (80% Variance)

	Using L1 Regularization with on 3-Hidden Layer Network with PCA (70% Variance) on varying Epochs

	Using L1 Regularization with on Deep-vs-Wide Network with PCA (70% Variance) 

	Using Deep vs Wide network with Dropout (PCA 70% Variance))



Conclusion:

As mentioned in the ‘No Free Lunch Theorem’ none of model optimization technique is expected to perform better than any other optimization algorithm. The performance of the model depends heavily on the architecture of the model and the type of the dataset used. For the MIT/TIDIGITS corpus dataset it can be observed that the L2 norm regularization technique has worked best with small network architecture thereby giving an accuracy of more than 96%. However, the model has performed the least if it is used for classification without any dimensionality reduction technique of the feature space. The model without PCA has classified the Test corpus dataset with an accuracy of only 9%. We can also conclude that the Dropout Technique did not give as great results as Lp Norm Regularization techniques, since it is expected to give better results for comparatively larger dataset than the MIT/TIDIGITS corpus data.
