## Naive Bayes Classifier

### I. Applications

The naive bayes is a simple technique used to classify examples using the** Bayes Theorem**. It is called "naive" since it assumes that all the features are **independent**, i.e all the features are contributing equally to the probability of the example being assigned to a given class.

Even if they are simple models, they perform very well in many classification tasks \(document classification, spam detection...\).

### II. Algorithm

Given an example $$x$$ with $$n$$ features $$x = (x_1, ... , x_n) $$, we want to assign a class $$C_k$$ to the example, given its features. Thus we look for $$p(C_k | x_1, ... , x_n)$$. Using the Bayes Theorem, we can compute this probability with the following formula:


$$
 p(C_k | x) = \frac{p(x | C_k) p(C_k)}{p(x)}
$$


$$p(x|C_k)$$ tells us that given the features of the example, how _likely_ it is that it belongs to the class $$C_k$$. Thus it is called the **likelihood**.

$$p(C_k)$$ is the **prior**, which represents the probability of an example to be assigned to the class $$C_k$$. It is the _belief_ about this class. it is calculated using the formula: $$ \frac{\text{number of class samples}}{\text{dataset size}}$$

$$p(x)$$ is called the **evidence**. Since it is constant given the input, it is often discarded. Thus we are only interested in the numerator.

The _assumptions made on the likelihood_ is what differentiates the naives bayes classifiers. The likelihood can be computed using different probability distributions.

* The** Gaussian Naive Bayes **assumes that the likelihood of the features follow the Gaussian Distribution, i.e:


$$
 p(x|C_k) = \frac{1}{\sqrt{2 \pi \sigma_{C_k}^2}}\text{exp}(-\frac{(x - \mu_{C_k})^2}{2 \sigma_{C_k}^2})
$$


with $$\mu_{C_k}$$being the mean of class $$C_k$$ and $$\sigma_{C_k}$$ being its standard deviation.

* The **Bernoulli Naive Bayes **uses the Bernoulli distribution :


$$
  p(x|C_k) = \prod_{i=1}^n p_{k_i}^{x_i}(1 - p_{k_i}^{x_i})^{(1 - x_i)}
$$


with $$p_{k_i}^{x_i}$$ the probability that the class $$C_i$$ generated $$x_i$$.

* The **Multinomial **classifier uses a Multinomial distribution. The distribution is parameterized by a feature vector $$\theta_C  = (\theta_{C_1}, ...,  \theta_{C_n})$$ for each class $$C_i$$ with $$\theta_{C_i}$$ the probability of feature $$x_i$$ appearing in a sample of class $$C_k$$ \(i.e $$p(x_i | C_k)$$\).

Thus, we can build our classifier by following these steps:

1. For each class $$C_k$$, compute the prior $$p(C_k)$$
2. Depending on the assumption you make about the distribution of the data, compute the likelihood $$p(x | C_k)$$ \(for example mean and std-deviation for the Gaussian Naive Bayes\)
3. Once you have the posterior for each class, keep the one with the highest probability. This is called the MAP decision rule \(Maximum a Posteriori\). It can be written $$\hat{y} = \underset{(1, ..., k)}{\text{argmax }} 
     p(x|C_k)p(C_k)$$

###### Limitations:

* Of course one of the biggest limitations of the Naive Bayes is the fact that it assumes that all your features are independant, which is not always the case. It is always good to run a pearson correlation test in order to find to which degree your features are correlated.
* It makes strong assumptions about the distribution of your data.



