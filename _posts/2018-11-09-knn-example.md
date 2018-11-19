---
layout: post
title: "KNN Example"
date: 2018-11-09 09:41:47
image: '/assets/img/'
description: 'Exploring KNN using a simple worked example and sk-learn'
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
- Intel AI Machine Leaning
twitter_text: 
---

# K Nearest Neighbors Example

Using the telecom customer churn data we will do some preprocessing and the use a KNN model to make some predictions. First we will import the .csv file and display its contents

{% highlight python %}
import os, pandas as pd

fileloc = 'Done/Intel-ML101-Class1/Intel-ML101_Class1/data/Orange_Telecom_Churn_Data.csv'
data = pd.read_csv(fileloc)

data.head(1).T
{% endhighlight %}
![Output](/assets/img/IntelAI/knn1.png){:class="img-responsive"}

# Pre-processing
Some features such as **state, area code, phone number** are not useful for the predictive model because they are just *identifiers*. We will remove them, so we are only left with useful features.

{% highlight python %}
data.drop(['state','area_code','phone_number'], axis=1, inplace=True)
data.columns
{% endhighlight %}
>Index(['account_length', 'intl_plan', 'voice_mail_plan',
       'number_vmail_messages', 'total_day_minutes', 'total_day_calls',
       'total_day_charge', 'total_eve_minutes', 'total_eve_calls',
       'total_eve_charge', 'total_night_minutes', 'total_night_calls',
       'total_night_charge', 'total_intl_minutes', 'total_intl_calls',
       'total_intl_charge', 'number_customer_service_calls', 'churned'],
      dtype='object')

Now we notice that some of the fields are categories['intl_plan', 'voice_mail_plan', 'churned']. We are going to numerically **encode** these features.

{% highlight python %}
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()

for col in ['intl_plan', 'voice_mail_plan', 'churned']:
    data[col] = lb.fit_transform(data[col])
{% endhighlight %}

### Feature Scaling methods
 - **Standard Scaler**: Mean center data and scale to unit **variance**
 - **Minimum-Maximum Scaler**: Scale data to fixed range (usually 0-1)
 - **Maximum Absolute Value Scaler**: Scale maximum absolute value

KNN models require scaled data, therefore we are going to scale the data using a **MinMaxScaler**

{% highlight python %}
from sklearn.preprocessing import MinMaxScaler

msc = MinMaxScaler()

data = pd.DataFrame(msc.fit_transform(data), columns=data.columns)
{% endhighlight %}

# Training
The data preprocessing is done. We will create two tables, one with all the features used to train the model and the other one with **churned**, this is the label.

{% highlight python %}
#The following is a list of all the features except churned
x_cols = [x for x in data.columns if x!= 'churned']

#split the data by selecting the relevant columns from the dataframe
X_data = data[x_cols]
y_data = data['churned']
{% endhighlight %}
The two tables can now be used to train the KNN Classifier Model. We are going to use a k-value of **3**.

{% highlight python %}
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn = knn.fit(X_data, y_data)

#After learning how does it classify the input data?
y_pred = knn.predict(X_data)
{% endhighlight %}

# Accuracy
Accuracy is a very simple way of measuring error. It just shows the percentage of correct labeles examples.

{% highlight python %}
def accuracy(real, predict):
    return sum(real == predict)/float(real.shape[0])
{% endhighlight %}

{% highlight python %}
print(accuracy(y_data,y_pred))
{% endhighlight %}
> 0.9422

# Distance for weighting
`KNeighborsClassifier` uses **uniform** for the weights. Let's try changing it to **distance**

{% highlight python %}
from sklearn.neighbors import KNeighborsClassifier

knn_dist = KNeighborsClassifier(n_neighbors=3, weights='distance')

knn_dist = knn_dist.fit(X_data, y_data)

#After learning how does it classify the input data?
y_pred_dist = knn_dist.predict(X_data)
{% endhighlight %}

{% highlight python %}
print(accuracy(y_data,y_pred_dist))
{% endhighlight %}
>1.0

# Manhattan Distance
Power parameter for the Minkowski metric. When p = 1, this is equivalent to using **manhattan_distance (l1)**, and **euclidean_distance (l2)** for p = 2. Default is p=2.

{% highlight python %}
from sklearn.neighbors import KNeighborsClassifier

knn_mink = KNeighborsClassifier(n_neighbors=3, p=1)

knn_mink = knn_mink.fit(X_data, y_data)

#After learning how does it classify the input data?
y_pred_mink = knn_mink.predict(X_data)
{% endhighlight %}

{% highlight python %}
print(accuracy(y_data,y_pred_mink))
{% endhighlight %}
>0.9456

Looks like distance weighting gives an accuracy of 100%. This is wrong because all of the training data was used to train the model and test the same data. The model is overfitting. For new examples it may perform poorly.

# Accuracy vs k-value

To test how k-values affect the accuracy we are going to plot the values for different k-vals.

{% highlight python %}
from sklearn.neighbors import KNeighborsClassifier

accuracy_vals = []

for i in range (1,20):
    knn_x = KNeighborsClassifier(n_neighbors=i, p=1)
    
    knn_x = knn_x.fit(X_data, y_data)

    y_pred_x = knn_x.predict(X_data)
    
    accuracy_vals.append(accuracy(y_data,y_pred_x))
{% endhighlight %}

{% highlight python %}
print(accuracy_vals)
import matplotlib.pyplot as pl, numpy as np

pl.plot(np.arange(1, len(accuracy_vals)+1), accuracy_vals, 'b', label = 'x', linewidth = 3)
{% endhighlight %}
> [1.0, 0.9286, 0.9456, 0.9214, 0.9356, 0.9216, 0.929, 0.917, 0.9256, 0.9126, 0.921, 0.9124, 0.9178, 0.9096, 0.9146, 0.9074, 0.9126, 0.9054, 0.9104]

![Output](/assets/img/IntelAI/knn2.png){:class="img-responsive"}

When k=1 it's neighbor is the same as itself. Therefore the model is again overfitting. As k-value increases the accuracy is generally getting worse as the model shits to underfitting.

Credits to [Intel AI Academy](https://software.intel.com/en-us/ai-academy)













