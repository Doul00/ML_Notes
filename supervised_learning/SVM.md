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
	<img src="/assets/svm/svm-margins.png" alt="Margins equations" height="300" width="400">
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
\min_{w} \frac{1}{2}w^Tw \\  \text{subject to }  y_i(w x_i + b) \ge 1 \\  \forall x_i \text{ in } i\text{ ,..., } n
$$


Solving this problem gives us the optimal hyperplane. One interesting thing to notice is that our problem is _quadratic_ (its surface is a paraboloid), which means that it has **one** global minimum. This way, SVMs avoid the problems encountered in Neural nets with local optimums.

###### Solving the optimization problem
In order to solve this problem, we will use the [Lagrangian multipliers](https://en.wikipedia.org/wiki/Lagrange_multiplier). Its goal is to minimize the objective function while integrating the constraint at the same time. For a visual explanation, you can check this [link](https://www.svm-tutorial.com/2016/09/duality-lagrange-multipliers/).

The Lagrangian is written:

$$
L(x, \lambda, \nu) = f(x) + \sum_{i=i}^N \lambda_i f_i(x) + \sum_{i=i}^N \nu_i h_i(x) \\ \text{with } \lambda_i \ge 0 \text{ and } \nu_i \ge 0
$$

with  $$f_i(x) \le 0$$ being the inequality constraints and $$h_i(x) = 0$$ being the equality ones.

In our case, the constraint is $$y_i(wx_i - b) \ge 1$$, thus it is written:

$$
L(w, b, \alpha) = \frac{1}{2} w^Tw - \sum_{i=1}^N \alpha_i y_i(wx_i - b) - 1 \\
L(w, b, \alpha) = \frac{1}{2} w^Tw - \sum_{i=1}^N \alpha_i y_i(wx_i - b) + \sum_{i=1}^N \alpha_i \text{ }(*)
$$

We reach the minimum when $$\nabla L(w, b, \alpha) = 0$$  i.e:

$$
\frac{\partial L}{\partial w} = w - \sum_{i=1}^N \alpha_i y_i x_i = 0 \\
\frac{\partial L}{\partial b} = \sum_{i=1}^N \alpha_i y_i  = 0 
$$

From the first equation we get: $$w = \sum_{i=1}^N \alpha_i y_i x_i$$, and from the second: $$\sum_{i=1}^N \alpha_i y_i  = 0$$
It means that our $$w$$ is a linear combination of the $$ \alpha_i$$s , the class labels and the points. $$\alpha_i$$ represents the **contribution** of point $$x_i$$ in calculating the value of $$w$$. As we saw before, the margins are defined by the points lying on them, called the support vectors. **Any point which is far from the margin does not define it, thus it does not influence the size of the margin**. It means that all those points will have **zero** contribution when calculating the value of $$w$$. All their $$\alpha_i$$ will be equal to zero.

By substituting $$w$$ by $$\sum_{i=1}^N \alpha_i y_i x_i$$ in $$(*)$$ and developing, we reach the result:

$$
L(w, b, \alpha) = \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{1 \le i,j \le N} \alpha_i \alpha_j y_i y_j (x_i \cdot x_j)
$$

We managed to get rid of the $$w$$ and $$b$$ parameters, we only have to maximize over the values of the $$\alpha$$ by respecting the constraint $$\sum_{i=1}^N \alpha_i y_i  = 0$$.


###### Soft Margin Classifier

As we already saw above, the SVM built by looking for the maximum margin is very sensitive to the support vectors.

<figure align="center">
	<img src="/assets/svm/svm-smc.png" alt="SVM margins" height="300" width="500">
	<figcaption>SVM Margins<a href="quantstart.com">(Source)</a></figcaption>
</figure>


As shown in the example above, adding a single data point can completely change the hyperplane, thus impact the predictive power of the SVM. To prevent this problem, one can use a_ soft margin classifier_ which allows for separation errors in order to prevent overfitting.

The intuition for this is that we would like to **move** our margins a little bit to allow them to classify points that are not on the correct side of the hyperplane (not respecting the condition $$wx - b \ge 1$$). However, we need to penalize those points using a value $$\xi_i$$ for each point $$x_i$$ not respecting the constraint.

Going from the equation $$y_i(wx_i - b) \ge 1 $$, our first hyperplane, $$H_1$$, will have the equation $$wx - b\ge 1 - \xi_i$$ (by replacing $$y_i$$ by 1) and the second one will have the equation $$wx - b \ge -1 + \xi_i$$.

Graphically, it gives something like this:

<figure align="center">
	<img src="/assets/svm/svm_soft_margins.png" alt="Moving the margins" height="300" width="300">
	<figcaption>Moving the margins</figcaption>
</figure>

* All the points falling on the good side of the hyperplane (but between the margin and the separator) will have $$0 \le \xi_i \le 1$$.
* If $$\xi_i = 1$$, it means that our margin has the same equation as the separator (i.e $$wx - b = 0$$).
* If $$\xi_i > 1$$, it means that the point is on the other side of the separator
* Obviously, a negative $$\xi_i$$ does not make sense since it means that our point is already well classified.

We indicate how strong we want our penalities to be by adding a parameter $$C$$. The higher the C, the stronger our penalities and the less our margins will move. We tend to get a hard-margin SVM in that case. However, the smaller the $$C$$, the more error we tolerate.

Our optimization problem becomes:

$$
\min_{w, b, \xi} \frac{1}{2}w^Tw + C(\sum_{i=1}^n \xi_i) \\
\text{subject to }  y_i(w x_i + b) \ge 1 - \xi_i \\  \forall x_i \text{ in } i\text{ ,..., } n\\
\xi_i \ge 0
$$

The Lagrangian is written:
$$
L(w, b, \xi, \alpha, \mu) = \frac{1}{2} w^Tw + C(\sum_{i=1}^N \xi_i) - \sum_{i=1}^N \alpha_i y_i(wx_i - b) + \sum_{i=1}^N \alpha_i - \\ \sum_{i=1} \xi_i \alpha_i - \sum_{i=1}^N \mu_i \xi_i

\\

L(w, b, \xi, \alpha, \mu) = \frac{1}{2} w^Tw + \sum_{i=1}^N \xi_i(C - \mu_i - \alpha_i) - \sum_{i=1}^N \alpha_i y_i(wx_i - b) + \sum_{i=1}^N \alpha_i \text{ } (**)
$$

We want to **maximize** $$(**)$$ with the constraints:

$$
\sum_{i=1}^N \alpha_i y_i = 0\\
0 \le \alpha_i \le C
$$

The optimization problem is solved using the [SMO Algorithm](https://en.wikipedia.org/wiki/Sequential_minimal_optimization). You can check out the link if you're curious about it.


##### II. When the data is not linearly separable

In order to solve problems where the data is not linearly separable, we move our data into higher dimensions. By projecting our data into a higher dimension, it is possible to find a hyperplane separating the value. To perform this "projection" operation, we use a **kernel function**.


