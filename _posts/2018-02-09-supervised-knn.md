---
layout: post
title: "K-nearest Neighbours"
date: 2018-02-09 11:19:47
image: '/assets/img/'
description: 'Exploring KNN, a simple supervised machine learning technique'
tags:
- python
- knn
- machine learning
- numpy
- scikit
- classification
- feature scaling
- regression
categories:
- ML & Data Mining 
twitter_text: 
---

# What is Machine Learning?
Machine Learning is using a computer to learn features and trends from data. Just like humans, a machine algorithm should increase its `Performance` through increased `Experience`, meaning it should get better with more data.

Machine Learning is already with us in many services we use:
- Spam Filtering
- Search engines
- Fraud/Anomaly detection
- Speech Recognition
- Self-Driving cars
- Recommendation Systems(Amazon/Netflix)

### There are two main types of machine learning:

**Supervised** - Data points have known outcome

**Unsupervised** - Data points have unknown outcome

We're going to focus on **Supervised** learning here, and it can be broken down into two more categories:

*`Regression`* - Outcome is continuous(numerical)

*`Classification`* - Outcome is a Class label


Basically, in supervised learning you input `Data with answers` which is used to `fit` a `MODEL`, which can then `predict answers` for a bunch of `data without answers`. Let's look at an example with Spam or not spam.
![Output](/assets/img/DataMiningSnippets/8.png){:class="img-responsive"}

### Machine Learning Vocabulary

- **Target** - predicted category or value of the data*(column to predict)*
- **Features** - properties of the data used for prediction*(non-target columns)*
- **Example** - a single data point within the data
- **Label** - the target value for a single data point

![Output](/assets/img/DataMiningSnippets/9.png){:class="img-responsive"}

# K- Nearest Neighbours

In pattern recognition, the k-nearest neighbours algorithm is a non-parametric method used for classification and regression. We are going to focus on classification.

## What is classification?
A flower shop wants to guess a customer's purchase from similarity to most recent purchase. Which flower is a customer based on similairity to previous purchases?


![Output](/assets/img/DataMiningSnippets/10.png){:class="img-responsive"}

We can classify the images to find the similarities in this case.

### What are needed for classification
- **Model data with:**
	- Features that can be quantitated
	- Labels that are known
- **Methods to measure similarity**

### KNN Classification
How can we use this particular algorithm for classification?
Let's say we have the data shown below, and we want to classify the new data point, what kind of metrics do we need to consider?
#### K-Neighbors
We would have to choose the correct value for `K Neighbors`, how many points around the new point to consider. The value of `K` affects the decision boundary. We won't be going into more depth here.
#### Distance to other points
We would also have to measure the closeness of the new point to other points, in this case, the new point is closer to the orange colour which may be significant.

![Output](/assets/img/DataMiningSnippets/11.png){:class="img-responsive"}

#### Measuring Distances
There are two ways to measure distances between two points. `Euclidean` distance(L2 Distance), and `Manhattan` distance(L1 or City Block Distance). Which one is better will depend on your dataset.

Scaling is also very important especially when there are many different features. Normalising the data in the preprocessing stage by performing a feature scaling stops "large" features from taking over too much predictive power. [Feature scaling](https://www.coursera.org/learn/machine-learning/lecture/xx3Da/gradient-descent-in-practice-i-feature-scaling) usually improves accuracy.
![Output](/assets/img/DataMiningSnippets/12.png){:class="img-responsive"}



## Feature Scaling

### Feature Scaling methods
 - **Standard Scaler**: Mean center data and scale to unit **variance**
 - **Minimum-Maximum Scaler**: Scale data to fixed range (usually 0-1)
 - **Maximum Absolute Value Scaler**: Scale maximum absolute value

### The Syntax
Make sure you have `scikit-learn` installed.
We will use `StandardScaler`, there is also `MinMaxScaler`, and `MaxAbsScaler` available.
{% highlight python %}
from sklearn.preprocessing import StandardScaler #Import the scaling method

StdSc = StandardScaler() #Create an instance of the class

StdSc = StdSc.fit(X_data) #Fit the scaling parameters
X_scaled = KNN.transform(X_data) #Transform the data
{% endhighlight %}


## KNN Modelling

### Characteristics
- Fast to create model because it simply stores data
- Slow to predict because many distance calculations
- Can require lots of memory if the data-set is large

### The Syntax
The `fit` and `predict`/`transform` syntax is common in machine learning. We could have also used `KNeighborsRegressor` to perform regression.
{% highlight python %}
from sklearn.neighbors import KNeighborsClassifier #Import the class method

KNN = KNeighborsClassifier(n_neighbors=3) #Create and instance of the class

KNN = KNN.fir(X_data, Y_data)
y_predict = KNN.predict(X_data)
{% endhighlight %}


Credits to [Intel AI Academy](https://software.intel.com/en-us/ai-academy)













