### Support Vector Machines

#### I. Applications

SVMs are one of the most powerful machine learning algorithms. They can perform linear classification as well as non-linear classification. Support Vector Machines can also be used for regression.

Basically, the goal of the SVM is to draw a hyperplane which will separate the data for classification. It does this by finding the best hyperplane, i.e the one with margins that allow a good separation of the data.

An excellent tutorial about the maths behind SVMs can be found here: [https://www.svm-tutorial.com/](https://www.svm-tutorial.com/). I will not go too deep into the maths.

#### II. Algorithm

##### I. When the data is linearly separable

###### Maximum Margin Classifier

When the data is linearly separable, the goal is to compute the optimal hyperplane separating the data. To find the hyperplane with the biggest margin, we first select two hyperplanes with no data points in between and we maximise the distance between them. Those two hyperplanes are delimited by the data points lying on the margin. Those points are called the **support vectors. **In the following example, points A, B and C are the support vectors. This example shows us that the classification power of the SVM **depends a lot on the position of the support vectors**. This kind of SVM are called _Maximum Margin Classifiers_**.**

![](/assets/support_vectors.png)

_Source: quantstart.com_

Let's define our hyperplanes' equations: we will have $$W^T x + b = \delta $$ for the first hyperplane, $$H_0$$, and $$w^T x + b = - \delta $$ for the second hyperplane, $$H_1$$. Let's also suppose that we have a binary classification problem, with two classes, i.e $$y(x) = 1$$ or $$y(x) = -1$$.

Thus, we want, for all $$x_i$$ in the data:


$$
W^T x_i + b \ge \delta \text{ or }  W^T x_i + b \le -\delta
$$


which means that each point will be either on one side of $$H_1$$ or on the other side of $$H_0$$

To classify a new example $$ x^* $$, we calculate $$W^T x^* + b = y^*$$ . If $$y^* \le -\delta $$ then the new example belongs to one class \(or to the other class if $$ y^* \ge \delta $$\).

![](/assets/svm.png)

If we multiply the two-sides of the hyperplane's equations by $$y_i$$ and combine the two equations, we find that, for all $$x_i$$:


$$
y_i(W^T x_i + b) \ge \delta
$$


and the margin can be computed as:


$$
m = \frac{2}{||w||}
$$


with $$w$$ being the normal unit vector to one of the hyperplanes. The margin will grow when the norm of $$w$$ will decrease. Thus maximizing the margin and minimizing the norm are the same things. Finally, we obtain the following optimization problem:


$$
\text{minimize in } (w, b) \text{ ; } \\ ||w|| \\  \text{subject to }  y_i(W^T x_i + b) \ge \delta \\  \forall x_i \text{ in } i\text{ ,..., } n
$$


Solving this problem gives us the optimal hyperplane.

###### Soft Margin Classifier

As we already saw above, the SVM built by looking for the maximum margin is very sensitive to the support vectors.

![](/assets/svm-smc.png)

_Source: quantstart.com_

As shown in the example above, adding a single data point can completely change the hyperplane, thus impact the predictive power of the SVM. To prevent this problem, one can use a_ soft margin classifier_**,  **which allows for separation errors in order to prevent overfitting. The number of misclassified elements is tuned via a parameter $$C$$ . A small value of C means low bias but high variance \(a few errors on the training set but we tend to overfit\) while a higher value of $$C$$ means higher bias and lower variance.

##### II. When the data is not linearly separable

In order to solve problems where the data is not linearly separable, we move our data into higher dimensions. By projecting our data into a higher dimension, it is possible to find a hyperplane separating the value. To perform this "projection" operation, we use a **kernel function**.

---

### 

[^1]: [Can be found here ](https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/#one)

