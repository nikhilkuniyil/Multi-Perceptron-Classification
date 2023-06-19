# Multi-Perceptron-Classification
## About
Implemented and trained a multi-perceptron classification algorithm from scratch to classify a distribution of points throughout the coordiante plane. The perceptrons
were first trained using the perceptron learning algorithm and with stochastic gradient descent, with a learning rate 0.001 and 500 epochs. Then, the MSE
(mean-squared error) loss function, $L = \frac{1}{2} (y - \hat{y})^2 $, was used, which also led to convergence. The hyperplanes with respect to the distribution
of points in the coordinate plane are graphed and both display convergence.

## Intuition
Any kind of data that is linearly separable will allow the perceptron to converge with the perceptron learning algorithm or the MSE loss function. However,
many iterations fo the perceptron learning algorithm will yield different weight values for the trained percetron, while the use of the MSE loss function 
will always yield the exact same weight values for the trained perceptron. This is because the MSE loss function, when graphed, produces a prabola with one
global minima, which is identified by our optimization function (stochastic gradient descent). Each time that our perceptron(s) is trained using the 
MSE loss function, SGD allows our perceptron to converge to the same excact global miminma of the function, meaning the weights will always be the same.

## Tools/Frameworks
• NumPy <br>
• Matplotlib <br>
• Jupyter Notebook <br>
