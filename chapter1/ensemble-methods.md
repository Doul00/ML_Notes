## Ensemble Methods

---

### I. Applications

Ensemble methods is the use of multiple machine learning algorithms in order to get better results. They combine multiple hypothesis to form a better one. The ensemble is itself a machine learning algorithm since it can be trained and then used to make predictions. There many methods used for ensemble learning, such as voting, bagging, stacking and boosting. Later, we will see the most used ensemble algorithms.

###### Voting\(classification\) & Averaging\(regression\):

In voting, the first step is to create multiple classification models using the training dataset. Once the models are trained, different methods are used to classify the test examples:

* **Majority voting:** In this case, every model will perform the classification on the example. The output will correspond to the one that receives more than 50% of the votes.
* **Weighted voting: **In this case, each model has a weight that will make its vote more or less important. It is useful to make the better models more important.
* **Simple averaging: **The average value of the output is used for the prediction.
* **Weighted averaging: **Same as the simple averaging, but each model has a weight.

###### Stacking:

The idea behind stacking is to train multiple algorithms on the dataset and to generate a new dataset using those algorithms. Then, a _combiner_ algorithm is used to make predictions on this new set.

###### Bagging \(bootstrap aggregating\):

In bagging, models are trained on random subsets of the original data. The sampling is done by replacement, which means an example might be repeated. After training, the models are combined using averaging or voting techniques. Bagging reduces variance and helps avoid overfitting.

###### Boosting:

The goal of boosting is to learn weak classifiers and to combine them into a strong learner. Every model has a weight assigned to it, indicating how it influences the final decision.

The algorithm works this way: Let's take $$n$$ models and mark them as $$h_1, ..., h_n$$

* First, $$h_1$$ is trained on a sub-sample of the data and makes predictions on the rest of the data
* The following step is then repeated for each model $$h_i$$:
  * Take the  errors from the previous model and assign more weight to them, so that the next model will try to predict them well.

  * Make new predictions using $$h_i$$
* When you reach the limit of $$n$$ models, combine them to create a stronger learner.

![](/assets/boosting.png)

_Source: Analytics Vidhya_



For instance, in this example, we want to build a classifier for pluses and minuses.

* We start with D1 with is trained and then performs predictions. As we can see, it misclassified 3 pluses as minuses.
* So before creating D2, we give a higher weight to those misclassified examples.
* We add D2 and we perform a new prediction. This time is classified the 3 pluses very well but misclassifies the red minuses.
* We repeat the algorithm for D3
* Finally, we  weight the three models and combine them to create our final classifier.

### II. Algorithms

#### I. Random Forest

The Random Forest algorithm is a bagging of multiple decision trees. It is a way of averaging the decision trees and preventing overfitting, while performing better classification/regression tasks.

###### Algorithm:

The training phase applies the principle of bagging. Each of the decision trees is trained on a sub-sample of the data. During the training phase, another sub-sampling is done when choosing the splits for the trees: A random sample of features are selected for the splits. By doing this, features which are important for the prediction will be represented in many trees, creating a correlation between them.After training, the prediction for the new examples can be made by averaging the results of the decision trees or by using a majority voting method.

This method performs well since it reduces the variance of the model, without increasing the bias.

#### II. Ada boost



