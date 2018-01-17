## Support Vector Machines

#### I. Applications

SVMs are one of the most powerful machine learning algorithms. They can perform linear classification as well as non-linear classification. Support Vector Machines can also be used for regression.

Basically, the goal of the SVM is to draw a hyperplane which will separate the data for classification. It does this by finding the best hyperplane, i.e the one with margins that allow a good separation of the data.

An excellent tutorial about the maths behind SVMs can be found here: [https://www.svm-tutorial.com/](https://www.svm-tutorial.com/).

#### II. Algorithm

##### I. When the data is linearly separable

###### Maximum Margin Classifier

When the data is linearly separable, the goal is to compute the optimal hyperplane separating the data. To find the hyperplane with the biggest margin, we first select two hyperplanes with no data points in between and we maximise the distance between them. Those two hyperplanes are delimited by the data points lying on the margin. Those points are called the **support vectors. **In the following example, points A, B and C are the support vectors. This example shows us that the classification power of the SVM **depends a lot on the position of the support vectors**. This kind of SVM are called _Maximum Margin Classifiers_**.**

<figure align="center">
	<img src="/assets/svm/support_vectors.png" alt="Support vectors" height="300" width="300">
	<figcaption>Support vectors<a href="quantstart.com">(Source)</a></figcaption>
</figure>


Let's define our hyperplanes' equations: we will have $$w x + b = \delta $$ for the first hyperplane, $$H_0$$, and $$w x + b = - \delta $$ for the second hyperplane, $$H_1$$ \(we will replace $$\delta$$ by 1 and -1 to simplify the problem\). Let's also suppose that we have a binary classification problem, with two classes, i.e $$y(x) = 1$$ or $$y(x) = -1$$.

Thus, we want, for all $$x_i$$ in the data:


$$
w x_i + b \ge 1 \text{ , } \forall y = 1 \\ \text{ or }  \\ w x_i + b \le -1 \text{ , } \forall y = -1
$$


which means that each point will be either on one side of $$H_1$$ or on the other side of $$H_0$$. For instance, in the picture below, all the red points are the points for which $$wx - b \ge 1 $$.

<figure align="center">
	<img src="/assets/svm/svm-margins.png" alt="Margins equations" height="400" width="400">
	<figcaption>Margins equations<a href="svm-tutorial.com">(Source)</a></figcaption>
</figure>


If we multiply the two-sides of the hyperplane's equations by $$y_i$$ and combine the two equations, we find that, for all $$x_i$$:

$$
y_i(w x_i + b) \ge 1
$$

Now, we need to find the hyperplane $$H_0$$ , with the equation $$wx - b = 0$$, that the distance between the two margins is as big as possible. Indeed, the bigger the distance between the two of them, the better the separation. But how do we compute this distance?


<figure align="center">
	<img src="/assets/svm/svm_margin_distance.png" alt="Margins distance" height="300" width="300">
	<figcaption>Distance between the two margins</figcaption>
</figure>

Let's write $$m$$ as the distance between the two margins. In order to get $$m$$, we need to project $$x_0$$ onto $$H_1$$. By translating $$x_0$$ using a vector of magnitude $$m$$ and perpendicular to $$H_1$$, we will be able to obtain this point.
Hopefully, we already know a vector perpendicular to $$H_1$$, witch is $$w$$. But $$w$$ does not have the good magnitude. Let's first normalize it:

$$
u = \frac{w}{||w||}
$$ 

Then we multiply it by $$m$$:

$$
k = mu \text{ and } ||k|| = m
$$

Thus we have $$x^{'}_0 = x_0 + k$$ with $$x^{'}_0$$ the point we were looking for. Since $$x^{'}_0$$ is on $$H_1$$, we have $$wx^{'}_0 - b = 1$$.
We replace $$x^{'}_0$$ by its value and we obtain:
$$
w(x_0 + k) - b = 1 \\
w(x_0 + m\frac{w}{||w||}) -b = 1 \\
wx_0 + m\frac{w \cdot w}{||w||} - b = 1 \\
wx_0 + m\frac{||w||^2}{||w||} - b = 1 \\
wx_0 -b + m||w|| = 1 \\
$$

We know that $$x_0 \in H_1$$ thus $$wx_0 - b = 1$$.
Finally, we get:

$$
m = \frac{2}{||w||}
$$

The margin will grow when the norm of $$w$$ will decrease. Thus maximizing the margin and minimizing the norm are the same things. Finally, we obtain the following optimization problem:


$$
\text{minimize in } (w, b) \text{ ; } \\ ||w|| \\  \text{subject to }  y_i(w x_i + b) \ge 1 \\  \forall x_i \text{ in } i\text{ ,..., } n
$$


Solving this problem gives us the optimal hyperplane.

###### Solving the optimization problem



###### Soft Margin Classifier

As we already saw above, the SVM built by looking for the maximum margin is very sensitive to the support vectors.

<figure align="center">
	<img src="/assets/svm/svm-smc.png" alt="SVM margins" height="300" width="300">
	<figcaption>SVM Margins<a href="quantstart.com">(Source)</a></figcaption>
</figure>


As shown in the example above, adding a single data point can completely change the hyperplane, thus impact the predictive power of the SVM. To prevent this problem, one can use a_ soft margin classifier_**,  **which allows for separation errors in order to prevent overfitting. The number of misclassified elements is tuned via a parameter $$C$$ .

##### II. When the data is not linearly separable

In order to solve problems where the data is not linearly separable, we move our data into higher dimensions. By projecting our data into a higher dimension, it is possible to find a hyperplane separating the value. To perform this "projection" operation, we use a **kernel function**.

---

###

[^1]: [Can be found here ](https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/#one)

