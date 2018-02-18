## Linear Discriminant Analysis

### I. Motivation

LDA is a **supervised** dimensionality reduction technique. In this way, it is similar to the PCA, but the latter is an unsupervised techniques, it does not need the class labels to perform the dimensionality reduction.

Its goal is to find the principal components which **maximize **the separation between the classes of the dataset and **minimize** the within those classes. The idea behind this is that we want to find the axes that give us the biggest difference between the classes's means \(the bigger the difference, the more the classes will be separated\), while taking in account the variance of the classes.

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

