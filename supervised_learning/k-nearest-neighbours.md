## K-Nearest Neighbours

#### I. Applications

The KNN algorithm can be used both for classification and regression purposes. In the case of classification, the output is a class membership. When used for regression, the output is a continuous value computed using average the neighbours's value.

![](/assets/knn.png)

_Source: coxdocs.org_

The "**K" **parameters indicates the number of neighbours chosen in order** **to determine the class of the unknown object.

#### II. Algorithm

The training phase consists in storing the feature vectors of each class. During the classification phase, the unknown feature vector is classified by taking the dominant class among its neighbours. Most of the time, the E_uclidian distance_ is used. However,  in case of text-classification for example, other distances are used, such as the _Hamming distance_

The greater of value of K, the less vulnerable the algorithm is to the noise. However, it is not very efficient if the data is skewed \(not evenly distributed\).

###### Weighted classifier:

One way to make the algorithm perform better is to weight the neighbours' participation by taking into account their distance to the unknown data point.

It is also good to apply dimensionality reduction techniques in order to prevent the curse of dimensionality when working in higher dimensions.

