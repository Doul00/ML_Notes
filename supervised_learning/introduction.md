# Supervised Learning

This section will cover Supervised Learning, which regroups machine learning techniques used with **labeled data**, i.e, for each example in the **training set**, we have the desired output \(**label**\).

A supervised learning algorithm is able to analyse the training data and to produce an inferred function, which is able to map new examples. These kind of algorithms can solve two problems:

* **Classification** problems, which consist in finding to which category a new example belongs. One common example is digit classification, i.e telling if a digit is a zero, a one, etc...
* **Regression** problem, which estimate the relationship between the data attributes in order to predict a continuous value. One example is house prices prediction using different features \(number of bedrooms, kitchen size ...\).

One of the biggest challenge of the supervised learning algorithms is to **generalize** well from the training data to the new situations. If the algorithm is not able to generalize, i.e it is only able to perform well on example it has already seen, it is said that the algorithm is **overfitting.** Otherwise, if the algorithm is not able to learn from the training set, it is said that it is **underfitting**.

When an algorithm is underfitting, we say that the algorithm is **biaised**. The **bias **represents the prediction errors made by the model. If the model has a high bias, it will perform badly because of many errors.

When an algorithm is overfitting, it will have a high variance. The **variance** represents the error caused by sensitivity to small changes in the training set. An algorithm with high variance will have a tendency to model the **noise **of the training data.

Finding the right balance between bias an variance is called the **bias-variance tradeoff**. 



