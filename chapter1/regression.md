## Regression

---

### Linear Regression

#### **I. Applications**

Linear regression is used to predict a variable which output is a continuous value. For instance, it can be used to predict the price of a house given some of its features. It performs well when the output variable $$y$$ is a linear combination of the features. Otherwise, it will perform poorly.

Its goal is to draw a line which fits the data in the best way possible.

![](/assets/linear_regression.png)

_Source: Wikipedia_

#### **II. Model and cost function**

We model the relationship between the feature vector $$Y$$ and the output variable $$Y$$ as follow: $$Y = \theta^T X + b$$, with $$\theta$$ being the weight matrix and $$b$$ the intercept. Those weights indicate to which extend each features participate in the final value. For instance, if a feature $$x^{(i)}$$ has a weight of 0, it does not participate at all in the final result.

##### **Hypothesis:**

Since we want a line which fits the data in the best way,  we need to have a function which describes the output in best way possible. This function \(written $$h$$\(x\) \) is called the _hypothesis._

##### **Cost:**

Our goal is to **minimize** the distance between the data points and the line itself. We want to find the best $$\theta$$ so that our hypothesis matches the output in the best way possible. In other words, we want to **maximize** the likelihood:


$$
L(\theta | X, Y) = P(Y|X, \theta) = \prod_{i = 1}^{n} P(y^{(i)} | x^{(i)}, \theta)
$$


Since we want to maximize the likelihood, we will choose the parameters that give us the highest probabilities:


$$
\hat{\theta} = \underset{\theta}{argmax}  
L(\theta|X, Y)
$$


This is the **Maximum Likelihood Estimator\(MLE\).** To prevent underflow, we will work with the log likelihood.

By defining $$p(y^{(i)}|x^{(i)}) = \mathcal{N}(\theta^T x^{(i)}, \sigma^2)$$,  applying the MLE formula and maximizing over $$\theta$$, we see that the result is the same as minimizing the mean squared error: \(for details please see [http://cs229.stanford.edu/notes/cs229-notes1.pdf\)](http://cs229.stanford.edu/notes/cs229-notes1.pdf)
$$
J(\theta) = \frac{1}{2} \sum_{i=1}^{n} (y^{(i)} - h_\theta^{(i)})^2
$$
**Optimization:**

###### Gradient descent:

One way to minimize this cost function is through gradient descent. We take small step with a rate $$\alpha$$ to update our weights.

First, we derive the cost function:


$$
\frac{\partial}{\partial{\theta_j}} J(\theta) = (Y - h_\theta)x_j
$$
And the update rule is:


$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial{\theta_j}} J(\theta)
$$


###### Normal equations:

Another way of minimizing the cost function is by using the normal equations. We look for a point where the gradient is equal to zero. Since the cost function is convex, this point will be a local minimum. Using matrix derivation, solving $$\nabla_\theta J(\theta) = 0$$ gives us

$$\theta = (X^T X)^{-1} X^T y$$

Again, take a look at Andrew Ng's notes for more details.

---

### Logistic Regression

#### I. Applications

Logistic regression is used for classification tasks. Its goal is to predict the output of a variable taking **discrete** values, for example 0 or 1.

#### II. Model and cost function

###### Model:

If we used linear regression for classification, it would perform poorly and give us probabilities higher than 1, which does not make any sense. This is why the logistic regression makes use of the **sigmoid function**: $$f(x) = \frac{1}{1 + e^{-x}}$$ , which is useful since it maps any real number to a value between 0 and 1.

###### Cost function:

Let


$$
h(x) = f(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}
$$


be our hypothesis function and let's assume that $$p(y = 1 | x) = h_\theta(x)$$ and $$p(y = 0|x) = 1 - h_\theta(x)$$ .

It can also be written: $$p(y|x, \theta) = (h_\theta(x))^y (1 - h_\theta(x))^{1 - y}$$.

Again, we compute the log-likehood and we still want to maximize its value.

###### Optimization:

We will maximize the log-likelihood using gradient **ascent **this time. Indeed, we are **maximizing** the log-likelihood. The update rule is: $$\theta_j = \theta_j + \alpha \nabla_\theta l(\theta)$$ and the final result, after computing the gradient, is:


$$
\theta_j := \theta_j + \alpha (y^{(i)} - h_\theta^{(i)}(x))x_j^{(i)}
$$




