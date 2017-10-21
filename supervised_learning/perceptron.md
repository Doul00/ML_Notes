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

Multilayer perceptrons \(MLP\) are \_artificial neural networks. \_They are used for situations where the data is not linearly separable. The MLPs are composed of at least three layers:

* An input layer, which receives the data
* A hidden layer, made of neurons
* An output layer which computes the prediction

A neuron is the base unit of an artificial neural network an is inspired from the brain's neurons. In the same way as biological neurons, they receive information from multiple sources and if the external signals reach a stimulation threshold, the neuron sends a signal to other neurons. We say that it was activated.

Artificial neurons work in the same way. They receive inputs from the input layer, sum them and outputs a value thanks to an activation function. Mathematically, the sum of inputs is written as $$h(x) = w \cdot x + b$$ and the activation function as $$z(x) = \sigma(h(x)) $$ . This activation is non-linear, it helps the neural network break the linearity and learn non-linear features. There exist many activation functions, and we will study them later.

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

* We apply the activation function $$\sigma$$: $$z_i(x) = \sigma(h_i(x))$$
* It can be written as $$h(x) = W \cdot X + b$$ 

We apply the same algorithm until we reach the output neurons.

###### Backpropagation:

The first time we compute the forward computation, the outputs won't match the desired results. It means that we have to update our weights and biases in order to make our neural network more efficient. This is done with the _backpropagation**. **_

The idea behind this is that we compute the error at the output neurons and we _propagate_ the error back into the hidden layers in order to change the weights, so that our next forward computation will bring us closer to the result. The error is defined by a cost function and we want to minimize our error, i.e find the minimum \(the global minimum cannot be found directly, so most of the time we just accept a local minima\). In this way, backpropagation is analogue to the gradient descend.

![](/assets/backprop.png)

Let's take the following neural network as an example. We have:

* An input layer with two inputs, $$x_1 \text{ and } x_2$$.
* A hidden layer with two neurons $$ h_1 \text{ and } h_2 $$.
* An output layer with two neurons $$  o_1 \text{ and } o_2 $$. We write their activation functions as $$a_1 \text{ and } a_2$$ so that $$o_1 = \sigma(a_1) \text{ and } o_2 = \sigma(a_2)$$
* $$W \text{ and } V $$ are the weight matrices linking the layers.

Here we define two error values using $$  y_1 \text{ and } y_2 $$, which are the desired outputs. We can write the error matrix as:

$$e = \begin{bmatrix} o_1 - y_1 \\ o_2 - y_2
\end{bmatrix} \text{ and define the cost E as: } E = \frac{1}{2}||e||^2 = \frac{1}{2}((o_1 - y_1)^2 + (o_2 - y_2)^2)$$

First, we need to compute the gradient of the error with respect to the outputs. This value will tell us how the error changes when there are small changes in the outputs. We will start by writing the equations for neuron $$o_1$$:

*  $$\text{Error gradient: }\frac{\partial E}{\partial o_1} = o_1 - y_1$$

Now that we have the error gradient for the output. We want to propagate it back to the parameters responsible for the computation of this output value and update them. We know that $$o_1 = \sigma(a_1) =  \sigma(z_1 v_{11} + z_2 v_{21} + bo_1)$$ with $$z_1 = \sigma(h_1(x))$$.

From this formula, we can see the weights we need to update: $$v_{11}, v_{21} \text{ and } bo_1 $$.

* $$ \text{Gradient for 
   } v_{11} :\frac{\partial E}{\partial v_{11}} = \frac{\partial E}{\partial o_1} \frac{\partial o_1}{\partial v_{11}} = (o_1 - y_1)\sigma'(a_1) z_1$$ 
* $$ \text{Gradient for 
   } bo_1 :\frac{\partial E}{\partial bo_1} = \frac{\partial E}{\partial o_1} \frac{\partial o_1}{\partial bo_1} = (o_1 - y_1)\sigma'(a_1)$$
* $$ \text{Gradient for 
   } v_{21} :\frac{\partial E}{\partial v_{21}} = \frac{\partial E}{\partial o_1} \frac{\partial o_1}{\partial v_{21}} = (o_1 - y_1)\sigma'(a_1) z_2$$$$$$

It is easy to notice the common part in those gradients: $$(o_1 - y_1)\sigma'(a_1)$$ . We can write it as $$\delta^2_1$$. By repeating the same operation for output $$o_2$$, we have $$\delta^2_2 = (o_2 - y_2) \sigma'(a_2)$$ , with $$\delta^l_j$$ meaning "Error signal in layer $$l$$ for neuron $$j$$". We can write:


$$
\delta^2 = \begin{bmatrix} (o_1 - y_1)\sigma'(a_1)\\ (o_2 - y_2)\sigma'(a_2)
\end{bmatrix}
$$


Now we have the gradients for the weights and the bias. Let's propagate the signal further back into the network to update the outputs from the previous layer's neurons. We have $$z_1 = \sigma(h_1(x)) = \sigma(w_{11} x_1 + w_{21} x_2 + bh_1) $$.  We repeat the same operation as previously:


$$
 \text{Gradient for 
 } w_{11} :\frac{\partial E}{\partial w_{11}} = \frac{\partial E}{\partial o_1} \frac{\partial o_1}{\partial w_{11}} + \frac{\partial E}{\partial o_2} \frac{\partial o_2}{\partial w_{11}}
$$
Here the formula is not the same. Indeed, $$w_{11} $$ was also used to compute $$o_2$$ \(through $$h_1$$ 's output\)$$$$, so we need to take in account the error coming from this output too.


$$
\frac{\partial o_1}{\partial w_{11}} = \frac{\partial o_1}{\partial z_1} \frac{\partial z_1}{\partial w_{11}} = v_{11} \sigma'(a_1) x_1 \sigma'(h_1)= \delta^2_1v_{11}x_1\sigma'(h1)
$$

$$
\frac{\partial o_2}{\partial w_{11}} = \frac{\partial o_2}{\partial z_1} \frac{\partial z_1}{\partial w_{11}} = v_{12} \sigma'(a_2) x_1 \sigma'(h_1)= \delta^2_2 v_{12}x_1\sigma'(h1)
$$
Then :


$$
\frac{\partial E}{\partial w_{11}} = (\delta^2_1 v_{11} + \delta^2_2 v_{12}) x_1 \sigma'(h_1)
$$
For the bias, we have the following:


$$
\frac{\partial E}{\partial bh_1} = \delta^2_1v_{11} + \delta^2_2 v_{12}
$$
Again we can notice in a common part in the derivatives. We write it as $$\delta^1_1$$ . Is it easy to keep going and write all the formulas down, but let's factorise everything:

* To compute the derivative of the bias of neuron $$j$$ in the layer $$l$$, we have: 
  $$
  \frac{\partial E}{\partial b^l_j} = \delta^l_j
  $$
* To compute the derivative of the weight mapping neuron $$i$$ of layer $$ l - 1$$ to neuron $$j$$ of layer $$l$$:
  $$
  \frac{\partial E}{\partial w^l_{ij}} = \delta^l_j  a^{l-1}_i
  $$
  Or in simple words, "Error signal of the current layer for neuron $$j$$ multiplicated by output of the neuron $$i$$ in the previous layer".
* To compute the error signal of layer $$l$$:


$$



$$




