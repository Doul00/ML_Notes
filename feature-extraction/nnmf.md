## Non-Negative Matrix Factorization \(NNMF\)

### I. Applications {#nnmf-motivation}

Matrices are a widely used data representation format. However, the more data, the more they grow. Being able to represent them with smaller matrices can be very interesting. This is the goal of the _Non-Negative Matrix Factorization_. This method factorises a matrix $$R \in \mathbb{R}^{n, m}$$  into two matrices $$P \in \mathbb{R}^{n, k}$$ and such $$Q \in \mathbb{R}^{m, k}$$ such that:


$$
 \hat{R} = P \cdot Q^T \approx R
$$


with $$\hat{R}$$ an approximation of $$ R $$.

This method is widely used in computer vision, audio signal processing and recommender systems. It is useful where the data contains many attributes that are not necessarily strong predictors or are ambiguous, and it allows to put in evidence patterns in the data.

In the case of recommender systems, all the users have a rating matrix $$R$$, which represent to which extend they like or dislike an item. However this matrix is very sparse, since most users only rate a few items. Using an item matrix $$Q$$, which represents the item's attributes and a user matrix $$P$$, which indicates the user interest in those attributes, matrix factorisation is able to capture the interaction between the user and the items.

**Why non-negative? **The non-negativity is useful in applications where the data is naturally non-negative \(such as text-mining\) and allows an easier interpretation \(word-count, pixel intensity ...\). It also leads to _sparse vectors_ in the resulting matrices, which has interesting properties.

### II. Algorithm

The goal of the NNMF is to reconstruct $$R$$ from the two other matrices. Thus we want our reconstruction to be as close as possible to the original matrix. Thus we want to minimize the following function:


$$
 \min_{P, Q} \sum_{(i, j) \in k } (r_{ui} - q_i p_j)^2
$$


with $$k$$ being the set of \($$u, i$$\) for which the rating is _known. _By fitting only on the new items, the model will update the $$r_{ui}$$ for which the rating is unknown. This will bring out a good estimation of the unknown ratings, which can be used to perform recommendations. The minimisation is usually done with a stochastic gradient descent algorithm.

As said above, this method returns sparse vectors, but this sparsity is not controlled. However, it is possible to choose the number of non-zero entries in a vector using the $$l_0 \text{norm}$$, which corresponds to the number of non-zero elements in a vector. Using this method complicates the optimization process. Indeed, the $$l_0$$ minimization is regarded as a NP-Hard problem. Some approximation algorithms exist, for instance the[ Matching pursuit algorithm ](https://en.wikipedia.org/wiki/Matching_pursuit).

