# Java ML library
Here I implemented my own simplistic Machine Learning training model from scratch that uses feed-forward fully-connected neural networks. At the moment the code only supports the neural network structure with a single hidden layer. I used ReLU activation for forward propagation and Stochastic Gradient Descent to update neural network parameters. The model was trained and tested on MNIST dataset of handwritten nubmers. The prediction accuracy lies between 87-91% which is not great when comparing to state-of-art ML packages like PyTorch, but also isn't that bad for a classic stochastic gradient descent optimizer. Use of regularization techniques and more advanced optimizers like Adam should further improve the prediction accuracy, but I won't be implementing it here for now.

The code was written purely for academic purposes and by the time of writing it I wasn't that much familiar with Java.


**What's next:**

-- Restructure the code to add suport of multi-layered network structures.

-- I need to turn this code into an actual package that could be used by other applications.

