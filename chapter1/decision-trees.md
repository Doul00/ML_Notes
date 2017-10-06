### Decision Trees

#### I. Applications

Decision Trees are used to classify objects. Each node represents an observation about the item and the tree's leaves are the class labels. Unlike linear models, they are able to map non-linear relationships. At each node, the tree splits the data into two or more homogenous sets \(which are heterogeneous between each other\) in order to be able to discriminate well. Moreover, they are useful for data exploration as they do not require much data cleaning. However, they tend to overfit very easily.

Trees where the output is a continuous value are called _Regression Trees._

![](/assets/decision_tree.png)

_Source: Wikipedia_

This example comes from the titanic dataset of Kaggle. In this example, the decision tree outputs the class of a passenger \(survived or died\). The decision is made by looking at each of the passenger's attributes \(sex, age, number of siblings/spouses\) and by comparing the current passenger's attributes, the class is chosen. This example shows that the critical point of this algorithm is the selection of **discriminant features and the conditions to apply.**

#### II. Algorithms

There are many algorithms used to build decision trees. Here, we will take a look at the **Gini Index** and the** Information Gain **methods. Other methods can be found [here](https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/#one).

###### Gini Index:

The Gini performs only **binary splits.** In this procedure, different split points are tried and tested using a cost function. The split with the lower value is selected \(since we want to minimize the cost\). Once the split is performed, the function is recursively called on each new node.

The Gini Index indicates how "pure" a node is, i.e to which extend the training data is mixed in a node. For instance, if the node contains only data points from one class, the node would be pure and it's cost equal to zero.

It can be computed using the formula: $$G = p^2 + q^2$$ with $$p$$ being the probability of having one class members in a node, and $$q = 1 - p$$.

The Gini score is computed for each sub-node and then weighted using the number of instances.

![](/assets/Screen Shot 2017-09-02 at 18.00.08.png)_Source: Analytics Vidhya_

In this example[^1], we for the Gender, we have:

* Gini for female split: $$0.2^2 + 0.8^2 = 0.68$$
* Gini for male split: $$0.65^2 + 0.35^2 = 0.55$$
* Weighted Gini value: $$0.68 * (10/30) + 0.55 * (20/30) = 0.59$$

For the split on class, the final  value is 0.51. Thus, we split on Gender.

###### Information Gain:

This method works by calculating the entropy for each node. If a node is homogeneous, the entropy is equal to zero and when it is equal to one in the worst case \(50% of class 1 in the node and 50% of class 2\). The entropy can be calculated using the formula:


$$
\text{Entropy = } -p log2 p - q log2 q
$$


The steps to calculate entropy for a split are:

* Calculate entropy for the parent node
* Calculate entropy for each individual node and calculate the weighted average entropy.
* Finally, choose the entropy which has the lowest value between parent nodes and other splits.

In our previous example, we have the following results:

* Entropy for parent node: $$-0.5log_2(0.5) - 0.5log_2(0.5) = 1$$. 
* Entropy for female split: $$-0.2log_2(0.2) - 0.8log_2(0.8) = 0.72$$
* Entropy for male split: $$-0.65log_2(0.65) -0.35log_2(0.35) = 0.93$$
* Weighted entropy of both sub nodes: $$0.86$$

###### Preventing overfit:

Decision Trees tend to overfit very quickly. In the worst case there can be a node for each training example, resulting in a 100% accuracy in the training set but bad performance with the test set.

One way to prevent this is by setting constraints on the trees:

* Maximum tree depth: Prevents the tree from generating too many sub-nodes and learning specific relations, resulting in an overfit.
* Minimum number of samples for splitting a node. The value shouldn't be too high, otherwise underfit can occur.
* Minimum number of samples for a leaf.
* Maximum number of features used for splits.
* Pruning the tree. Nodes which do not provide much discriminative information are removed. More [here.](https://en.wikipedia.org/wiki/Pruning_%28decision_trees%29)



