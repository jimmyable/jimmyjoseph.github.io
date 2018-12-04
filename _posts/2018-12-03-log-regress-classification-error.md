---
layout: post
title: "Logistic Regression and Classification Error Metrics"
date: 2018-12-03 10:28:47
image: '/assets/img/'
description: 'In this example we look at correlations, startified shuffle split, and error metrics produced from different cross validation methods. We also look at when we remove highly correlated columns.'
tags:
- python
- logistic regression
- machine learning
- numpy
- scikit
- classification
- ridge regression
- error metrics
- lasso regression
- stratified shuffle split
categories:
- Intel AI Machine Leaning
twitter_text: 
---

# Logistic Regression and Classification Error Metrics

In this worked example, we use [Human Activity Recognition with Smartphones](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) database, which was built from the recordings of study participants performing activities of daily living (ADL) while carrying a smartphone with an embedded inertial sensors. The objective is to classify activities into one of the six activities (walking, walking upstairs, walking downstairs, sitting, standing, and laying) performed.

For each record in the dataset it is provided: 

- Triaxial acceleration from the accelerometer (total acceleration) and the estimated body acceleration. 
- Triaxial Angular velocity from the gyroscope. 
- A 561-feature vector with time and frequency domain variables. 
- Its activity label. 

Now, lets import the dataset.

{% highlight python %}
import pandas as pd, numpy as np

data = pd.read_csv('/data/Human_Activity_Recognition_Using_Smartphones_Data.csv')
{% endhighlight %}

The data columns are all floats except for the activity label.

{% highlight python %}
data.dtypes.value_counts()
{% endhighlight %}
> - float64    561
- object       1
- dtype: int64

{% highlight python %}
data.dtypes.tail()
{% endhighlight %}

> - angle(tBodyGyroJerkMean,gravityMean)    float64
- angle(X,gravityMean)                    float64
- angle(Y,gravityMean)                    float64
- angle(Z,gravityMean)                    float64
- Activity                                 object
- dtype: object

The data are all scaled from -1 (minimum) to 1.0 (maximum).

{% highlight python %}
data.iloc[:, :-1].min().value_counts()
{% endhighlight %}

> - -1.0    561
- dtype: int64

{% highlight python %}
data.iloc[:, :-1].max().value_counts()
{% endhighlight %}

> - 1.0    561
- dtype: int64

Examine the breakdown of activities--they are relatively balanced.

{% highlight python %}
data.Activity.value_counts()
{% endhighlight %}

> - LAYING                1944
- STANDING              1906
- SITTING               1777
- WALKING               1722
- WALKING_UPSTAIRS      1544
- WALKING_DOWNSTAIRS    1406
- Name: Activity, dtype: int64

Since Scikit-learn doesn't accept sparse matrix for the prediction columns we will use `LabelEncoder` to convert activity labels into integers. Lets try and look at the values.

{% highlight python %}
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['Activity'] = le.fit_transform(data.Activity)
data['Activity'].sample(5)
{% endhighlight %}

> - 5944    0
- 3473    1
- 3114    1
- 4753    1
- 2959    0
- Name: Activity, dtype: int64

# Correlations

We are going to calculate the correlations between the dependent variables.

{% highlight python %}
# Calculate the correlation values
feature_cols = data.columns[:-1]
corr_values = data[feature_cols].corr()

#Simplify by emptying all the data below the diagonal
tril_index = np.tril_indices_from(corr_values)

#Make the unused values NaNs
for coord in zip(*tril_index):
    corr_values.iloc[coord[0], coord[1]] = np.NaN

#Stack the data and convert to a data frame
corr_values = (corr_values.stack().to_frame().reset_index().rename(columns={'level_0':'feature1','level1':'feature2',0:'correlation'}))

#Get the absolute values for sorting
corr_values['abs_correlation'] = corr_values.correlation.abs()
{% endhighlight %}

Visualising the absolute value correlations using a histogram.

{% highlight python %}
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('talk')
sns.set_style('white')
sns.set_palette('dark')

ax = corr_values.abs_correlation.hist(bins=50)

ax.set(xlabel='Absolute Correlation', ylabel='Frequency');
{% endhighlight %}

![Output](/assets/img/IntelAI/logError1.png){:class="img-responsive"}

The most highly correlated values:

{% highlight python %}
corr_values.sort_values('correlation', ascending=False).query('abs_correlation>0.8')
{% endhighlight %}

![Output](/assets/img/IntelAI/logError2.png){:class="img-responsive"}

# Stratified Shuffle Split
We will now split the data into test and train sets. There are lots of methods available but Scikit-learn's `StratifiedShuffleSplit` is used now. This type of split maintains the same ratio of predictor classes. 

{% highlight python %}
from sklearn.model_selection import StratifiedShuffleSplit

# Get the split indices
# Test size - 30%
strat_shuf_split = StratifiedShuffleSplit(n_splits=1,test_size=0.3, random_state=42)

train_idx, test_idx = next(strat_shuf_split.split(data[feature_cols], data.Activity))

# Create the dataframes
X_train = data.loc[train_idx, feature_cols]
y_train = data.loc[train_idx, 'Activity']

X_test  = data.loc[test_idx, feature_cols]
y_test  = data.loc[test_idx, 'Activity']
{% endhighlight %}

Regardless of methods used to split the data, compare the ratio of classes in both the train and test splits.

{% highlight python %}
y_train.value_counts(normalize=True)
{% endhighlight %}

>- 0    0.188792
- 2    0.185046
- 1    0.172562
- 3    0.167152
- 5    0.149951
- 4    0.136496
- Name: Activity, dtype: float64

{% highlight python %}
y_test.value_counts(normalize=True)
{% endhighlight %}

>- 0    0.188673
- 2    0.185113
- 1    0.172492
- 3    0.167314
- 5    0.149838
- 4    0.136570
- Name: Activity, dtype: float64

# Logistic Regression
We are going to fit multi-class model using all the features with no regularization. 

{% highlight python %}
from sklearn.linear_model import LogisticRegression

# Standard logistic regression
lr = LogisticRegression().fit(X_train, y_train)
{% endhighlight %}

Using cross validation to determine the hyperparameters, fit models using **L1**, and **L2** regularization. Store each of these models as well. Note the limitations on multi-class models, solvers, and regularizations. The regularized models, in particular the L1 model, will probably take a while to fit.

Different solvers that can be used are:
- `liblinear`
- `newton-cg`
- `lbfgs`
- `sag`
- `saga`

{% highlight python %}
from sklearn.linear_model import LogisticRegressionCV

# L1 regularized logistic regression
lr_l1 = LogisticRegressionCV(Cs=10, cv=4, penalty='l1', solver='liblinear').fit(X_train, y_train)

# L2 regularized logistic regression
lr_l2 = LogisticRegressionCV(Cs=10, cv=4, penalty='l2').fit(X_train, y_train)

{% endhighlight %}

# Coefficients
For each of the models lets compare the magnitude of the coefficients.

{% highlight python %}
# Combine all the coefficients into a dataframe
coefficients = list()

coeff_labels = ['lr', 'l1', 'l2']
coeff_models = [lr, lr_l1, lr_l2]

for lab,mod in zip(coeff_labels, coeff_models):
    coeffs = mod.coef_
    coeff_label = pd.MultiIndex(levels=[[lab], [0,1,2,3,4,5]], 
                                 labels=[[0,0,0,0,0,0], [0,1,2,3,4,5]])
    coefficients.append(pd.DataFrame(coeffs.T, columns=coeff_label))

coefficients = pd.concat(coefficients, axis=1)

coefficients.head()
{% endhighlight %}

![Output](/assets/img/IntelAI/logError3.png){:class="img-responsive"}

Now let's plot six seperate plots for each of the multi-class coefficients.

{% highlight python %}
fig, axList = plt.subplots(nrows=3, ncols=2)
axList = axList.flatten()
fig.set_size_inches(10,10)


for ax in enumerate(axList):
    loc = ax[0]
    ax = ax[1]
    
    data = coefficients.xs(loc, level=1, axis=1)
    data.plot(marker='o', ls='', ms=4.0, ax=ax, legend=False)
    
    if ax is axList[0]:
        ax.legend(loc=4)
        
    ax.set(title='Coefficient Set '+str(loc))

plt.tight_layout()
{% endhighlight %}

# Probability Prediction
Lets predict and store the class for each model. We will also store the probability for the predictions of classes for each model.

{% highlight python %}
# Predict the class and the probability for each

y_pred = list()
y_prob = list()

coeff_labels = ['lr', 'l1', 'l2']
coeff_models = [lr, lr_l1, lr_l2]

for lab,mod in zip(coeff_labels, coeff_models):
    y_pred.append(pd.Series(mod.predict(X_test), name=lab))
    y_prob.append(pd.Series(mod.predict_proba(X_test).max(axis=1), name=lab))
    
y_pred = pd.concat(y_pred, axis=1)
y_prob = pd.concat(y_prob, axis=1)

y_pred.head()
{% endhighlight %}

![Output](/assets/img/IntelAI/logError4.png){:class="img-responsive"}

{% highlight python %}
# Predict the class and the probability for each

y_pred = list()
y_prob = list()

coeff_labels = ['lr', 'l1', 'l2']
coeff_models = [lr, lr_l1, lr_l2]

for lab,mod in zip(coeff_labels, coeff_models):
    y_pred.append(pd.Series(mod.predict(X_test), name=lab))
    y_prob.append(pd.Series(mod.predict_proba(X_test).max(axis=1), name=lab))
    
y_pred = pd.concat(y_pred, axis=1)
y_prob = pd.concat(y_prob, axis=1)

y_pred.head()
{% endhighlight %}

![Output](/assets/img/IntelAI/logError5.png){:class="img-responsive"}

# Error metrics
Lets now calculate error metrics for each model. We will need to combine multi-class metrics into a single value for each model.

* accuracy
* precision
* recall
* fscore
* confusion matrix

{% highlight python %}
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

metrics = list()
cm = dict()

for lab in coeff_labels:

    # Preciision, recall, f-score from the multi-class support function
    precision, recall, fscore, _ = score(y_test, y_pred[lab], average='weighted')
    
    # The usual way to calculate accuracy
    accuracy = accuracy_score(y_test, y_pred[lab])
    
    # ROC-AUC scores can be calculated by binarizing the data
    auc = roc_auc_score(label_binarize(y_test, classes=[0,1,2,3,4,5]),
              label_binarize(y_pred[lab], classes=[0,1,2,3,4,5]), 
              average='weighted')
    
    # Last, the confusion matrix
    cm[lab] = confusion_matrix(y_test, y_pred[lab])
    
    metrics.append(pd.Series({'precision':precision, 'recall':recall, 
                              'fscore':fscore, 'accuracy':accuracy,
                              'auc':auc}, 
                             name=lab))

metrics = pd.concat(metrics, axis=1)

metrics
{% endhighlight %}

![Output](/assets/img/IntelAI/logError6.png){:class="img-responsive"}

# Confusion Matrix
Plotting the confusion matric for each model.

{% highlight python %}
fig, axList = plt.subplots(nrows=2, ncols=2)
axList = axList.flatten()
fig.set_size_inches(12, 10)

axList[-1].axis('off')

for ax,lab in zip(axList[:-1], coeff_labels):
    sns.heatmap(cm[lab], ax=ax, annot=True, fmt='d');
    ax.set(title=lab);
    
plt.tight_layout()
{% endhighlight %}

![Output](/assets/img/IntelAI/logError7.png){:class="img-responsive"}

# Removing Highly Correlated columns
Identifying and then removing the highly correlated columns.

Correlated features in general don't improve models. For linear models (e.g., linear regression or logistic regression), multicolinearity can yield solutions that are wildly varying and possibly numerically unstable. [Read More...](https://datascience.stackexchange.com/questions/24452/in-supervised-learning-why-is-it-bad-to-have-correlated-features)

{% highlight python %}
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold

#threshold with .7

sel = VarianceThreshold(threshold=(.7 * (1 - .7)))

data2 = pd.concat([X_train,X_test])
data_new = pd.DataFrame(sel.fit_transform(data2))


data_y = pd.concat([y_train,y_test])

from sklearn.model_selection import train_test_split

X_new,X_test_new = train_test_split(data_new)
Y_new,Y_test_new = train_test_split(data_y)
{% endhighlight %}

We are going to repeat the model building using the data with removed correlated columns.

Using **L1** and **L2** logistic regression.

{% highlight python %}
from sklearn.linear_model import LogisticRegression

# Standard logistic regression
lr = LogisticRegression().fit(X_new, Y_new)

from sklearn.linear_model import LogisticRegressionCV

# L1 regularized logistic regression
lr_l1 = LogisticRegressionCV(Cs=10, cv=4, penalty='l1', solver='liblinear').fit(X_new, Y_new)

# L2 regularized logistic regression
lr_l2 = LogisticRegressionCV(Cs=10, cv=4, penalty='l2').fit(X_new, Y_new)
{% endhighlight %}

# Comparing Coefficients
Compare the magnitudes of the coefficients for each of the models. If one-vs-rest fitting was used, each set of coefficients can be plotted separately. 

{% highlight python %}
# Combine all the coefficients into a dataframe
coefficients.sample(10)
coefficients = list()

coeff_labels = ['lr', 'l1', 'l2']
coeff_models = [lr, lr_l1, lr_l2]

for lab,mod in zip(coeff_labels, coeff_models):
    coeffs = mod.coef_
    coeff_label = pd.MultiIndex(levels=[[lab], [0,1,2,3,4,5]], 
                                 labels=[[0,0,0,0,0,0], [0,1,2,3,4,5]])
    coefficients.append(pd.DataFrame(coeffs.T, columns=coeff_label))

coefficients = pd.concat(coefficients, axis=1)

coefficients.head()
{% endhighlight %}

![Output](/assets/img/IntelAI/logError8.png){:class="img-responsive"}

# Plotting the coefficients
Let's look at the plots for the each of the multi-class coefficients.

{% highlight python %}
fig, axList = plt.subplots(nrows=3, ncols=2)
axList = axList.flatten()
fig.set_size_inches(10,10)


for ax in enumerate(axList):
    loc = ax[0]
    ax = ax[1]
    
    data = coefficients.xs(loc, level=1, axis=1)
    data.plot(marker='o', ls='', ms=4.0, ax=ax, legend=False)
    
    if ax is axList[0]:
        ax.legend(loc=4)
        
    ax.set(title='Coefficient Set '+str(loc))

plt.tight_layout()
{% endhighlight %}

![Output](/assets/img/IntelAI/logError9.png){:class="img-responsive"}

The points are much more consolidated this time around.

# Prediction of class probability
We are now going to predict and store the class for each model.
We will also store the probability for predicted classes for each model.

{% highlight python %}
# Predict the class and the probability for each

y_pred = list()
y_prob = list()

coeff_labels = ['lr', 'l1', 'l2']
coeff_models = [lr, lr_l1, lr_l2]

for lab,mod in zip(coeff_labels, coeff_models):
    y_pred.append(pd.Series(mod.predict(X_test_new), name=lab))
    y_prob.append(pd.Series(mod.predict_proba(X_test_new).max(axis=1), name=lab))
    
y_pred = pd.concat(y_pred, axis=1)
y_prob = pd.concat(y_prob, axis=1)

y_pred.head()

{% endhighlight %}

![Output](/assets/img/IntelAI/logError10.png){:class="img-responsive"}

{% highlight python %}
y_prob.head()
{% endhighlight %}

![Output](/assets/img/IntelAI/logError11.png){:class="img-responsive"}

# Error Metrics

For each model, calculate the following error metrics: 

* accuracy
* precision
* recall
* fscore
* confusion matrix

Decide how to combine the multi-class metrics into a single value for each model.

{% highlight python %}
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

metrics = list()
cm = dict()

for lab in coeff_labels:

    # Preciision, recall, f-score from the multi-class support function
    precision, recall, fscore, _ = score(Y_test_new, y_pred[lab], average='weighted')
    
    # The usual way to calculate accuracy
    accuracy = accuracy_score(Y_test_new, y_pred[lab])
    
    # ROC-AUC scores can be calculated by binarizing the data
    auc = roc_auc_score(label_binarize(Y_test_new, classes=[0,1,2,3,4,5]),
              label_binarize(y_pred[lab], classes=[0,1,2,3,4,5]), 
              average='weighted')
    
    # Last, the confusion matrix
    cm[lab] = confusion_matrix(Y_test_new, y_pred[lab])
    
    metrics.append(pd.Series({'precision':precision, 'recall':recall, 
                              'fscore':fscore, 'accuracy':accuracy,
                              'auc':auc}, 
                             name=lab))

metrics = pd.concat(metrics, axis=1)

metrics
{% endhighlight %}

![Output](/assets/img/IntelAI/logError12.png){:class="img-responsive"}

# Confusion Matrix
Finally the confusion matrix for the new models.

{% highlight python %}

fig, axList = plt.subplots(nrows=2, ncols=2)
axList = axList.flatten()
fig.set_size_inches(12, 10)

axList[-1].axis('off')

for ax,lab in zip(axList[:-1], coeff_labels):
    sns.heatmap(cm[lab], ax=ax, annot=True, fmt='d');
    ax.set(title=lab);
    
plt.tight_layout()

{% endhighlight %}

![Output](/assets/img/IntelAI/logError13.png){:class="img-responsive"}

Credits to [Intel AI Academy](https://software.intel.com/en-us/ai-academy)

{% highlight python %}

{% endhighlight %}










