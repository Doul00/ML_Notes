## Non-Negative Matrix Factorization

### I. Motivation {#nnmf-motivation}

Matrices are a widely used data representation format. However, the more data, the more they grow. Being able to represent them with smaller matrices can be very interesting. This is the goal of the _Non-Negative Matrix Factorization_. This method factorises a matrix $$R \in \mathbb{R}^{n, m}$$  into two matrices $$P \in \mathbb{R}^{n, k}$$ and such $$Q \in \mathbb{R}^{m, k}$$ such that:


$$
 \hat{R} = P \cdot Q^T \approx R
$$


with $$\hat{R}$$ an approximation of $$ R $$.

This method is widely used in computer vision, audio signal processing and recommender systems. It is useful where the data contains many attributes that are not necessarily strong predictors or are ambiguous, and it allows to put in evidence patterns in the data.

**Why non-negative? **The non-negativity is useful in applications where the data is naturally non-negative \(such as text-mining\) and allows an easier interpretation \(word-count, pixel intensity ...\). It also leads to _sparse vectors_ in the resulting matrices, which has interesting properties.

### II. Algorithm

The goal of the NNMF is to reconstruct $$R$$ from the two other matrices. Thus we want our reconstruction to be as close as possible to the original matrix. Thus we want to minimize the following function:


$$
 \min_{P, Q > 0} || R -  P \cdot Q^T ||^2_F
$$


with $$ || . ||^2_F $$ being the [Frobenius Norm. ](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm)

As said above, this method returns sparse vectors, but this sparsity is not controlled. However, it is possible to choose the number of non-zero entries in a vector using the $$l_0 \text{norm}$$, which corresponds to the number of non-zero elements in a vector. Using this method complicates the optimization process. Indeed, the $$l_0$$ minimization is regarded as a NP-Hard problem. Some approximation algorithms exist, for instance the[ Matching pursuit algorithm ](https://en.wikipedia.org/wiki/Matching_pursuit).

