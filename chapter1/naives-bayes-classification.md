## Classification

---

### Naive Bayes

---

### Decision Trees

#### I. Applications

Decision Trees are used to classify objects. Each node represents an observation about the item and the tree's leaves are the class labels. Unlike linear models, they are able to map non-linear relationships. At each node, the tree splits the data into two or more homogenous sets \(which are the heterogeneous between each other\) in order to be able to discriminate well. Moreover, they are useful for data exploration as they do not require much data cleaning. However, they tend to overfit very easily.

Trees where the output is a continuous value are called _Regression Trees._

![](/assets/decision_tree.png)

This example comes from the titanic dataset of Kaggle[^1]. In this example, the decision tree outputs the class of a passenger \(survived or died\). The decision is made by looking at each of the passenger's attributes \(sex, age, number of siblings/spouses\) and by comparing the current passenger's attributes, the class is chosen. This example shows that the critical point of this algorithm is the selection of **discriminant features and the conditions to apply.**

#### II. Algorithms

There are many algorithms used to build decision trees. Here, we will take a look at the **Gini Index** and the** Information Gain **methods.

###### Gini Index:

The Gini performs only **binary splits.** In this procedure, different split points are tried and tested using a cost function. The split with the lower value is selected \(since we want to minimize the cost\).

The Gini Index indicates how "pure" a node is, i.e to which extend the training data is mixed in a node. For instance, if the node contains only data points from one class, the node would be pure and it's cost equal to zero.

It can be computed using the formula: $$G = 1 - (p_1^2 + p_2^2)$$

###### Information Gain:



---

### K-Nearest Neighbours

#### I. Applications

The KNN algorithm can be used both for classification and regression purposes. In the case of classification, the output is a class membership. When used for regression, the output is a continuous value computed using average the neighbours's value.

![](/assets/knn.png)

The "**K" **parameters indicates the number of neighbours chosen in order** **to determine the class of the unknown object.

#### II. Algorithm

The training phase consists in storing the feature vectors of each class. During the classification phase, the unknown feature vector is classified by taking the dominant class among its neighbours. Most of the time, the E_uclidian distance_ is used. However,  in case of text-classification for example, other distances are used, such as the _Hamming distance_

The greater of value of K, the less vulnerable the algorithm is to the noise. However, it is not very efficient if the data is skewed \(not evenly distributed\).

###### Weighted classifier:

One way to make the algorithm perform better is to weight the neighbours' participation by taking into account their distance to the unknown data point.

It is also good to apply dimensionality reduction techniques in order to prevent the curse of dimensionality when working in higher dimensions.

