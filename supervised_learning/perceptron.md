### Perceptrons

#### I. Applications

The idea behind the perceptron is to separate the data using an hyperplane in order to classify the new examples. The hyperplane's equation has the form $$ f(x) = \sum_{i=1}^n w_i \cdot x_i + b = w \cdot x + b$$ with $$w$$ being a vector of weights and $$b$$ representing the bias, i.e the vertical offset of our hyperplan. For the sake of simplicity, we will consider from now on that the bias is already added into the weight matrix, i.e $$w_0 = b $$.

![](/assets/perceptron.png)

_Source: wp.wwu.edu_

We could then apply a simple method to classify a point. If it is above the line, the point belongs to class 1. Otherwise it belongs to class 2. It basically depends on the sign of $$f(x)$$. This can be written as:


$$
h(x) = 1 \text{ if } f(x) > 0 \text{ ; } h(x) = 1 \text{ otherwise}
$$


#### II. Algorithm

The training of the perceptron is online: Instead of looking at all the dataset, it looks at one example at a time, processes it, and then moves the to next example. Most of the time, the mean squared error is chosen as a loss function.

For each example $$x_i$$ in the dataset, we perform the following steps:

* Compute the output $$y_i$$ for the example with $$y_i = w.x_i$$
* If the output matches the desired output $$d_i$$, continue
* Otherwise, update the weights: $$w_i = w_i + (d_i - y_i)x_i$$
* Repeat until the loss function reaches a pre-defined threshold, or when the maximum number of iterations  is reached.

![](/assets/perceptron_train.png)

_Source: sebastianraschka.com_

The main problem of the single layer perceptron is that it cannot solve problems **when the data is not linearly separable**. Indeed, we can see that $$f(x)$$ is a linear combination. This is why we use _multilayer perceptrons._

### Multilayer perceptrons

#### I. Applications

Multilayer perceptrons \(MLP\) are _artificial neural networks. _They are used for situations where the data is not linearly separable. The MLPs are composed of at least three layers:

* An input layer, which receives the data
* A hidden layer, made of neurons
* An output layer which computes the prediction

A neuron is the base unit of an artificial neural network an is inspired from the brain's neurons. In the same way as biological neurons, they receive information from multiple sources and if the external signals reach a stimulation threshold, the neuron sends a signal to other neurons. We say that it was activated.

Artificial neurons work in the same way. They receive inputs from the input layer, sum them and outputs a value thanks to an activation function. Mathematically, the sum of inputs is written as $$h(x) = w \cdot x + b$$ and the activation function as $$z(x) = \phi(h(x)) $$ . This activation is non-linear, it helps the neural network break the linearity and learn non-linear features. There exist many activation functions, and we will study them later.

#### II. Algorithm

The learning algorithm for the MLPs is the following:

* Random initialization
* Forward pass
* Compute loss
* Backpropagation and parameters update

###### Initialization:

The weights linking the outputs of neurons in one layer to the input of neurons in another layer must be initialized randomly in a MLP. Indeed, if they are all equal to 0, the MLP won't be able to learn since the gradient values computed during backpropagtion would be nullified.

###### Forward pass:

![](/assets/MLP.png)

How does the MLP computes a value given inputs?

Let $$X$$ be our feature vector of size $$n$$ and $$W$$ be our weight matrix. For each neuron $$i$$ in the hidden layer:

* We compute it's output function: $$h_i(x) = \sum_{i=1}^n x_i w_{ij} + b_i$$_  _

_with _$$w{ij}$$  being the weight mapping the output \(input in this case since we only have 1 hidden layer\) $$i$$ of the previous layer  to the neuron $$j$$, with $$b_i$$ being the bias for that neuron.

* We apply the activation function $$\phi$$: $$z_i(x) = \phi(h_i(x))$$
* It can be written as $$h(x) = W \cdot X + b$$ 

We apply the same algorithm until we reach the output neurons.

###### Backpropagation:

The first time we compute the forward computation, the outputs won't match the desired results. It means that we have to update our weights and biases in order to make our neural network more efficient. This is done with the _backpropagation**. **_

The idea behind this is that we compute the error at the output neurons and we _propagate_ the error back into the hidden layers in order to change the weights, so that our next forward computation will bring us closer to the result. The error is defined by a cost function and we want to minimize our error, i.e find the minimum \(the global minimum cannot be found directly, so most of the time we just accept a local minima\). In this way, backpropagation is analogue to the gradient descend.

We define 

