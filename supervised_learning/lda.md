## Linear Discriminant Analysis

### I. Motivation

LDA is a **supervised** dimensionality reduction technique. In this way, it is similar to the PCA, but the latter is an unsupervised techniques, it does not need the class labels to perform the dimensionality reduction.

Its goal is to find the principal components which **maximize **the separation between the classes of the dataset and **minimize** the within those classes. The idea behind this is that we want to find the axes that give us the biggest difference between the classes's means \(the bigger the difference, the more the classes will be separated\), while taking in account the variance of the classes.

<figure align="center">
	<img src="/assets/lda/fisher-lda.jpg" alt="Separating classes using LDA" height="400" width="400">
	<figcaption>Separating classes using LDA<a href="http://www.stackoverflow.com">(Source)</a></figcaption>
</figure>

### II. Algorithm

In order to perform the dimensionality reduction, we will maximize the following function


$$
 J(w) = \frac{w^T S_B w}{w^T S_W w}
$$


with $$S_B$$ being the **between-class **scatter matrix and $$S_W$$ being the **within-class** scatter matrix. The scatter matrices are an _estimate_ of the covariance matrix. From the formula, we can see that maximizing $$J(w)$$ makes the difference between the class bigger and their variances smaller, which corresponds to what we ultimately want.


$$
S_B = \sum_c (\mu_c - \bar{x})(\mu_c - \bar{x})^T \\ S_W = \sum_c \sum_{i \in c} (x_i - \mu_c)(x_i - \mu_c)^T
$$


with $$\bar{x}$$ the overall mean of the dataset and $$\mu_c$$ the mean for class $$C$$.

Now, in order to get the maximum, we derive $$J(w)$$ and we set it equal to zero:
$$
\frac{\partial J(w)}{\partial w} = \frac{(w^T S_W w)\frac{w^T S_B w}{\partial w} - (w^T S_B w)\frac{w^T S_W w}{\partial w}}{(w^T S_W w)^2} = 0\\
\frac{(w^T S_W w)(2 S_B w) - (w^T S_B w)(2 S_W w )}{(w^T S_W w)^2} = 0
$$

Then we need to solve the equation for the numerator equal to zero:

$$
(w^T S_W w)(2 S_B w) - (w^T S_B w)(2 S_W w) = 0 \\
$$

We divide by $$w S_W w$$:
$$
S_B w - \frac{w_T S_B w}{w^T S_W w}(S_W w) = 0 \\
S_B w - J(S_W w) = 0 \\
S_B w = J(S_W w)\\
$$

We obtain a generalized eigenvalue problem. Solving $$S_W^{-1}S_Bw = Jw$$ gives the the desired result.

To sum up, the steps of the algorithm are the following:

1. Compute the overall mean of the dataset
2. Compute the between-class and within-class scatter matrices
3. Solve the eigenvalue problem
4. Keep the $$k$$ eigenvectors with the highest eigenvalues
5. Project the data using those eigenvectors

