## Independent Component Analysis \(ICA\)

### I. Motivation

Most of the time, the usefulness of the ICA is explained using the "cocktail-party" problem. Imagine you have 3 people talking in a room and 3 microphones spread in that room, capturing what the people say. When recovering the signal from each microphone, you will have a mix of sounds, which will prevent you from clearly hearing what the people said. Using _ICA_, you will be able to recover the signals from each of the speakers.

Basically, the ICA is applied on multivariate signals and decomposes it into several components. The multivariate signals are supposed to be **linear** mixtures of **latent** variables, and this mixing is unknown. Those latent variables are called the In_dependent Components_ \(or sources\). Certain assumptions are made for the ICA about those sources:

* The extracted components must be **independant **from one another.
* All of the sources must not have Gaussian distributions.

It is important to note that if you have $$N$$ sources, you must have at least $$N$$ observations in order to recover them all.

### II. Algorithm

Let $$x = x_1, x_2, ..., x_n$$ be our observed data and $$ s = s_1, s_2, ..., s_n $$ our independent components. The goal is to find a weight  matrix \(also called unmixing matrix\) $$W$$ such that:


$$
 s = Wx
$$


###### Whitening:

The first step of the algorithm consists in whitening the data. The goal of the whitening is to transform a set of variables, $$X$$, with a known covariance matrix, written $$M$$, into a set of variables $$Y$$ with a covariance matrix equal to the identity matrix. It means that all the variables will be _uncorrelated _ and have a variance of 1. It is written:


$$
 Y = WX \text{ with } W^TW = M^{-1}
$$


Most of the time whitening is done using PCA.

Visually, it gives the following:

Imagine we have a cloud of data points:
<figure align="center">
	<img src="/assets/ica_original.png" alt="A transformation" height="200" width="200">
	<figcaption>Cloud of points<a href="http://arnauddelorme.com/ica_for_dummies/">(Source)</a></figcaption>
</figure>

We mix two clouds and obtain the following figure:

<figure align="center">
	<img src="/assets/ica_mixed.png" alt="A transformation" height="200" width="200">
	<figcaption>Cloud of points<a href="http://arnauddelorme.com/ica_for_dummies/">(Source)</a></figcaption>
</figure>

After whitening, we get the following cloud:

<figure align="center">
	<img src="/assets/ica_whitened.png" alt="A transformation" height="200" width="200">
	<figcaption>Cloud of points<a href="http://arnauddelorme.com/ica_for_dummies/">(Source)</a></figcaption>
</figure>


We can see that the variance is the same on both axis. The ICA algorithm will then just "rotate" the cloud back to its original representation.



