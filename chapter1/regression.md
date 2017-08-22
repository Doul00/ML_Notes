## Regression

### 

### Linear Regression

**I. Applications**

Linear regression is used to predict a variable which output is a real value. For instance, it can be used to predict the price of a house given some of its features. It performs well when the output variable $$y$$ is a linear combination of the features. Otherwise, it will perform poorly.

Its goal is to draw a line which fits the best the data.

![](/assets/linear_regression.png)



**II. Model and cost function**

We model the relationship between the feature vector $$y$$ and the output variable $$y$$ as follow: $$y = Wx + b$$, with $$W $$ being the weight matrix. Those weights indicate to which extend each of the features participate in the final value. For instance, if a feature $$x_i$$ has a weight of 0, it does not participate at all in the value.

Since we want a line which fits the data in the best way, our goal is to minimize the distance between the data points and the line itself. One way to do it is by computing the value $$(\hat{y} - y)^2$$ with $$\hat{y}$$ representing our estimation \(a point on the line\) point and $$y$$ being an actual data point. This is the squared error. Applying this to all the data, we obtain the **MSE** function \(Mean Squared Error\) : 





