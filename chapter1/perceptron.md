### Perceptrons

#### I. Applications

The idea behind the perceptron is to separate the data using an hyperplane in order to classify the new examples. The hyperplane's equation has the form $$ f(x) = \sum_{i=1}^n w_i .x_i + b = w . x + b$$ with $$w$$ being a vector of weights and $$b$$ representing the bias, i.e the vertical offset of our line. For the sake of simplicity, we will consider from now on that the bias is already added into the weight matrix, i.e $$w_0 = b $$.

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
* Repeat until the loss function reaches a pre-defined threshold, or when the iteration of epochs is reached.

![](/assets/perceptron_train.png)

_Source: sebastianraschka.com_

The main problem of the single layer perceptron is that it cannot solve problems **when the data is not linearly separable**. This is why we use _multilayer perceptrons._

### Multilayer perceptrons



