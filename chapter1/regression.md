## Regression

---

### Linear Regression

#### **I. Applications**

Linear regression is used to predict a variable which output is a continuous. For instance, it can be used to predict the price of a house given some of its features. It performs well when the output variable $$y$$ is a linear combination of the features. Otherwise, it will perform poorly.

Its goal is to draw a line which fits the best the data.

![](/assets/linear_regression.png)

#### **II. Model and cost function**

We model the relationship between the feature vector $$y$$ and the output variable $$y$$ as follow: $$y = \theta^T x + b$$, with $$\theta$$ being the weight matrix and $$b$$ the intercept. Those weights indicate to which extend each of the features participate in the final value. For instance, if a feature $$x^{(i)}$$ has a weight of 0, it does not participate at all in the value.

##### **Hypothesis:**

Since we want a line which fits the data in the best way,  we need to have a function which describes the output in best way possible. This function \(written $$\hat{y}$$\(x\) \) is called the _hypothesis_

##### **Cost:**

Our goal is to minimize the distance between the data points and the line itself. One way to do it is by computing the value $$(\hat{y}^{(i)} - y^{(i)})^2$$ with $$\hat{y}^{(i)}$$ representing our estimation \(a point on the line\) point and $$y^{(i)}$$ being an actual data point. This is the squared error. Applying this to all the data, we obtain the our cost function, $$J(\theta)$$ :

$$\frac{1}{n} \sum_{i=1}^{n} (\hat{y}^{(i)} - y^{(i)})^2$$**  **

##### **Optimization:**

###### Gradient descent:

One way to minimize this cost function is through _gradient descent. _We take small step with a rate $$\alpha$$ to update our weights.

First, we derive the cost function:

$$$$$$\frac{\partial}{\partial{\theta_j}} J(\theta) = (\hat{y}(x) - y)x_j$$

And the update rule is:

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial{\theta_j}} J(\theta)$$

###### Normal equations:

Another way of minimizing the cost function is by using the normal equations. We look for a point where the gradient is equal to zero. Since the cost function is convex, this point will be a local minimum. Using calculus, solving $$\nabla_\theta J(\theta) = 0$$ gives us 

$$\theta = (X^T X)^{-1} X^T y$$

---

### Logistic Regression





