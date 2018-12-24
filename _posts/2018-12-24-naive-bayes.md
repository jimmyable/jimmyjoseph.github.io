---
layout: post
title: "Naive Bayes"
date: 2018-12-24 18:11:47
image: '/assets/img/'
description: 'A quick and simple machine learning model, Naive Bayes'
tags:
- python
- naive bayes
- machine learning
- bayes
- scikit
- classification
- probability
- error metrics
- confusion matrix
- train test split
categories:
- Intel AI Machine Leaning
twitter_text: 
---

# Naive Bayes

Probability of a single event occuring, and both occuring(joint) can be shown as a venn diagriam below.

![Output](/assets/img/IntelAI/NB1.png){:class="img-responsive"}

What if we only get **P(Y)** and want to predict the **P(X`|`Y)**? This is represented below as conditional probability.

![Output](/assets/img/IntelAI/NB2.png){:class="img-responsive"}

The joint probability can be caculated then as follows:

![Output](/assets/img/IntelAI/NB3.png){:class="img-responsive"}

Finally, **Bayes** theorem can be derived from the conditional and join relationship as:

![Output](/assets/img/IntelAI/NB4.png){:class="img-responsive"}

Furthermore, Bayes theorem can be written as:

![Output](/assets/img/IntelAI/NB5.png){:class="img-responsive"}

But this is not **Naive Bayes**. While training if we calculate the joint probabilities by expanding all the features it would be difficult. If *C* is *class*, and *X* is *features* then,

![Output](/assets/img/IntelAI/NB6.png){:class="img-responsive"}


## The Naive Assumption
If we assume all features are independant of each other then the calculation becomes easier. This is the **Naive** assumption.

![Output](/assets/img/IntelAI/NB7.png){:class="img-responsive"}

When training Naive Bayes, the class assignment is selected based on *maximum a posteriori* (MAP) rule:

![Output](/assets/img/IntelAI/NB8.png){:class="img-responsive"}
This means, the class with the largest value is potetially selected.

We will be using the iris dataset to learn Naive Bayes.

{% highlight python %}
import pandas as pd, numpy as np
#loading the dataset
data = pd.read_csv("data/Iris_Data.csv")
{% endhighlight %}

A quick look at the different datatypes involved.

{% highlight python %}
data.dtypes
{% endhighlight %}

> - sepal_length    float64	
- sepal_width     float64
- petal_length    float64
- petal_width     float64
- species          object
- *dtype: object*

Look at the skew values and decide if any transformations need to be applied. Let's use skew value 0.75 as a threshold.

**Data skew primarily refers to a non-uniform distribution in a dataset**

{% highlight python %}
skew = pd.DataFrame(data.skew())
skew.columns = ['skew']
skew['too_skewed'] = skew['skew'] > .75
skew
{% endhighlight %}

![Output](/assets/img/IntelAI/NB10.png){:class="img-responsive"}
Fields are not too badly skewed.

Use `sns.pairplot` to plot the pairwise correlations and histograms. Use `hue="species"` as a keyword argument in order to see the distribution of species.

{% highlight python %}
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.pairplot(data, hue='species')
{% endhighlight %}

![Output](/assets/img/IntelAI/NB11.png){:class="img-responsive"}


# Gaussian Naive Bayes

We will now fit a Naive Bayes classifier to this data to predict "species".

Different types of NB available:
![Output](/assets/img/IntelAI/NB9.png){:class="img-responsive"}

Since iris data is continous, we will be using **GaussianNB**

We will be using `cross_val_score` to see how well our choice works.

{% highlight python %}
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

X = data[data.columns[:-1]] #all data excpet last column
y = data.species

GNB = GaussianNB()
cv_N = 4 #Determines the cross-validation splitting strategy. 4-kold fold now
scores = cross_val_score(GNB, X, y, n_jobs=cv_N, cv=cv_N)
print(scores)
np.mean(scores)
{% endhighlight %}

> [0.94871795 0.94871795 0.91666667 1.        ]
0.953525641025641

# Compaing Naive Bayes'

Lets compare cross validation scores for all types of Naive Bayes.

{% highlight python %}
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

X = data[data.columns[:-1]]
y = data.species
nb = {'gaussian': GaussianNB(),
     'bernoulli': BernoulliNB(),
     'mutinominal': MultinomialNB()}

scores = {}

for key, model in nb.items():
    s = cross_val_score(model, X, y, cv=cv_N, n_jobs=cv_N, scoring='accuracy')
    scores[key] = np.mean(s)
scores
{% endhighlight %}

>{'bernoulli': 0.3333333333333333,
 'gaussian': 0.953525641025641,
 'mutinominal': 0.9529914529914529}

# Why is bernoulli performing bad?

Looks like BernoulliNB results are very bad, but MultinomialNB is doing a very good job.

BernoulliNB is usually good for binary classification. Since we have three classes it doesn't work very well.

# Removing very predictive features

Looking at the pairplot histograms, we can see that the `petal`_features are very predictive so let's remove and see how it affects the results.

{% highlight python %}
X = data[['sepal_length', 'sepal_width']]
y = data.species

nb = {'gaussian': GaussianNB(),
     'bernoulli': BernoulliNB(),
     'mutinominal': MultinomialNB()}

scores = {}

for key, model in nb.items():
    s = cross_val_score(model, X, y, cv=cv_N, n_jobs=cv_N, scoring='accuracy')
    scores[key] = np.mean(s)
scores

{% endhighlight %}
>{'bernoulli': 0.3333333333333333,
 'gaussian': 0.7879273504273504,
 'mutinominal': 0.6800213675213675}

The accuracy scores for Gaussian and Multinominal have fallen. Whilst Bernoulli remains unaffected.

The Gaussian model seems to produce better accuracy scores, even with the removal of very predictive features.

# Limitations of Naive Assumptions
What happens if we push the naive assumption too much?

> We will create **0, 1 ,3 ,5 ,10 ,50 ,100** copies of `sepal_length` and fit a `GaussianNB` for each one.
* Keep track of the save the average `cross_val_score`.
* Create a plot of the saved scores over the number of copies.

{% highlight python %}
X = data[data.columns[:-1]]
y = data.species

n_copies = [0, 1, 3, 5, 10, 50, 100]


def create_copies_sepal_length(X, n):
    X_new = X.copy()
    for i in range(n):
        X_new['sepal_length_copy%s' % i] = X['sepal_length']
    return X_new


def get_cross_val_score(n):
    X_new = create_copies_sepal_length(X, n)
    scores = cross_val_score(GaussianNB(), X_new, y, cv=cv_N, n_jobs=cv_N)
    return np.mean(scores)


avg_scores = pd.Series(
    [get_cross_val_score(n) for n in n_copies],
    index=n_copies)

ax = avg_scores.plot()
ax.set(
    xlabel='number of extra copies of "sepal_length"',
    ylabel='average accuracy score',
    title='Decline in Naive Bayes performance');
{% endhighlight %}

![Output](/assets/img/IntelAI/NB12.png){:class="img-responsive"}

Naive Bayes assumes each feature is an independant variable, so we created mulitiple copies of one feature `sepal_length`. We see that as the data gets duplicated, the average accuracy scores diminishes.

# Naive Bayes on Human Activity Recongnition

First we load the data.

{% highlight python %}
data = pd.read_csv('data/Human_Activity_Recognition_Using_Smartphones_Data.csv')
{% endhighlight %}

Now let's look at the datatypes involved.

{% highlight python %}
data.dtypes
{% endhighlight %}

> - ...
- angle(tBodyAccJerkMean),gravityMean)    float64
- angle(tBodyGyroMean,gravityMean)        float64
- angle(tBodyGyroJerkMean,gravityMean)    float64
- angle(X,gravityMean)                    float64
- angle(Y,gravityMean)                    float64
- angle(Z,gravityMean)                    float64
- Activity                                 object
- Length: 562, dtype: object

Every feature is a `float64`, and the targets are objects.

Now let's create `X` and `y` from the dataset.


{% highlight python %}
X = data[data.columns[:-1]]
y = data.Activity
{% endhighlight %}

Creating training and test splits.

{% highlight python %}
from sklearn.model_selection import train_test_split

y_col = 'Activity'

feature_cols = [x for x in data.columns if x != y_col]
X_data = data[feature_cols]
y_data = data[y_col]

X_train, X_test, y_train, y_test = train_test_split(X_data,y_data,
                                                   test_size=0.3,
                                                   random_state=42)
{% endhighlight %}

Fitting `GaussianNB` to the training split. 

{% highlight python %}
from sklearn.naive_bayes import GaussianNB

GNB = GaussianNB()
GNB.fit(X_train, y_train)
y_pred = GNB.predict(X_test)

{% endhighlight %}

Creating a confusion matrix for the predictions. 

{% highlight python %}
import seaborn as sns

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

metrics = list()
cm = dict()
# Preciision, recall, f-score from the multi-class support function
precision, recall, fscore, _ = score(y_test, y_pred, average='weighted')
    
# The usual way to calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# ROC-AUC scores can be calculated by binarizing the data
auc = roc_auc_score(label_binarize(y_test, classes=[0,1,2,3,4,5]),
          label_binarize(y_pred, classes=[0,1,2,3,4,5]), 
          average='weighted')

# Last, the confusion matrix
cm = confusion_matrix(y_test, y_pred)

metrics.append(pd.Series({'precision':precision, 'recall':recall, 
                          'fscore':fscore, 'accuracy':accuracy,
                          'auc':auc}, 
                         name='GaussianNB'))

metrics = pd.concat(metrics, axis=1)

metrics
{% endhighlight %}

![Output](/assets/img/IntelAI/NB13.png){:class="img-responsive"}

{% highlight python %}
import matplotlib.pyplot as plt

axList = axList.flatten()
fig.set_size_inches(15, 15)

axList[-1].axis('off')

plot = sns.heatmap(cm, annot=True, fmt='d');
    
plt.tight_layout()

{% endhighlight %}

![Output](/assets/img/IntelAI/NB14.png){:class="img-responsive"}

# Discretization
Data discretization is defined as a process of converting continuous data attribute values into a finite set of intervals with minimal loss of information.

Now, let's discretize the Human activity dataset. There are many ways to do this, but we'll use `pd.DataFrame.rank(pct=True)`.

{% highlight python %}
data = data.rank(pct=True);
X_discrete = data[data.columns[:-1]].rank(pct=True)
X_discrete = X_discrete.rank(pct=True)

y = data.Activity
{% endhighlight %}

Look at the values. They are still not discrete. Modify `X_discrete` so that it is indeed discrete. (Hint: try to get the first 2 digits using `.applymap`)

{% highlight python %}
func = lambda x: '%.2f' % x
X_discrete = X_discrete.applymap(func)
{% endhighlight %}

Split `X_discrete` and `y` into training and test datasets.

{% highlight python %}
X_train, X_test, y_train, y_test = train_test_split(X_discrete,y_data,
                                                   test_size=0.3,
                                                   random_state=42)
{% endhighlight %}

Fit a MultinomialNB to the training split. Get predictions on the test set.

{% highlight python %}
from sklearn.naive_bayes import MultinomialNB

MNB = MultinomialNB()
MNB.fit(X_train, y_train)
y_pred = MNB.predict(X_test)

{% endhighlight %}

Plot the confusion matrix for predictions.

{% highlight python %}
import seaborn as sns

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

metrics = list()
cm = dict()
# Preciision, recall, f-score from the multi-class support function
precision, recall, fscore, _ = score(y_test, y_pred, average='weighted')
    
# The usual way to calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# ROC-AUC scores can be calculated by binarizing the data
auc = roc_auc_score(label_binarize(y_test, classes=[0,1,2,3,4,5]),
          label_binarize(y_pred, classes=[0,1,2,3,4,5]), 
          average='weighted')

# Last, the confusion matrix
cm = confusion_matrix(y_test, y_pred)

metrics.append(pd.Series({'precision':precision, 'recall':recall, 
                          'fscore':fscore, 'accuracy':accuracy,
                          'auc':auc}, 
                         name='GaussianNB'))

metrics = pd.concat(metrics, axis=1)

metrics
{% endhighlight %}

![Output](/assets/img/IntelAI/NB15.png){:class="img-responsive"}

{% highlight python %}
import matplotlib.pyplot as plt

axList = axList.flatten()
fig.set_size_inches(15, 15)

axList[-1].axis('off')

plot = sns.heatmap(cm, annot=True, fmt='d');
    
plt.tight_layout()
{% endhighlight %}

![Output](/assets/img/IntelAI/NB16.png){:class="img-responsive"}

Credits to [Intel AI Academy](https://software.intel.com/en-us/ai-academy)

{% highlight python %}

{% endhighlight %}










