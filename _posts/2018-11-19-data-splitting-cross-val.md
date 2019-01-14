---
layout: post
title: "Data Splitting & Cross Validation"
date: 2018-11-19 04:55:47
image: '/assets/img/'
description: 'How to split data into test and training set, and perform cross validation'
tags:
- python
- knn
- machine learning
- numpy
- scikit
- cross validation
- splitting
- scaling
categories:
- Intel AI Machine Leaning
twitter_text: 
---

# Splitting and Cross Validation

If we use all the available data to train and test the model then the model is overfitting the data. This means the model is really good at passing the test stage because it has seen all the examples beforehand.

The aim of ML is to create a generalised model, which can be tested on unseen data. To do this we split the available data into a train and test datasets. 

Furthermore we can perform cross-validation, which is essentially shuffling training and test data to see the effects on the errors produced.

- **Underfitting** - training and cross validation errors are high 
- **Overfitting** - training error is low, cross validation is high
- **Just Right** - training and cross validation errors are low

We can explore these concepts using the Ames, Iowa housing prices dataset.


{% highlight python %}
import os, pandas as pd, numpy as np

filepath = 'data/Ames_Housing_Sales.csv'
data = pd.read_csv(filepath)

print(data.shape)
{% endhighlight %}
>(1379, 80)

This is a large dataset, with 79 features + SalePrice(target) and 1379 examples. We are going to see what what datatypes (**object, float64, int64**) the features use.

{% highlight python %}
data.dtypes.value_counts()
{% endhighlight %}
> - object     43
- float64    21
- int64      16
- dtype: int64

# Dealing with Categories
During preprocessing, we want to make sure the data is normalised by using feature scaling. For numerical features we can scale using **StandardScaler** and **MinmaxScaling**.

But for Nominal values such as *True/False*, we will have to encode them, using **One hot encoding**. This can be done by converting them into 0's and 1's.

For Oridinal values such as categories ordered features we have to use **Ordinal encoding**.

In this case there are 43 (String)objects that need to be encoded.


{% highlight python %}
#select which columns contains objects
string_cols = data.dtypes == np.object
categorical_cols = data.columns[string_cols]
#print(string_cols)
{% endhighlight %}

Determine how many total features would be present, relative to what currently exists, if all string (object) features are one-hot encoded. Recall that the total number of one-hot encoded columns is `n-1`, where `n` is the number of categories.

{% highlight python %}
#Number of extra columns that would be created
num_ohc_cols = (data[categorical_cols]
               .apply(lambda x: x.nunique())
               .sort_values(ascending=False))

#If only one value, don't encode
small_num_ohc_cols = num_ohc_cols.loc[num_ohc_cols>1]

#the number of one-hot columns is n-1
small_num_ohc_cols -= 1

#Number of columns after ohcs
small_num_ohc_cols.sum()
{% endhighlight %}
> 215

# Encoding
We are going to use one hot coding on the above categorical features. 

- We'll make a copy of the **dataframe**
- Then one-hot encode each of the appropriate columns and add it back to the dataframe
- Dropping the existing String Categorical columns

{% highlight python %}
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import warnings
warnings.filterwarnings('ignore', module='sklearn')

# Copy of the data
data_ohc = data.copy()

# The encoders
le = LabelEncoder()
ohc = OneHotEncoder()

for col in num_ohc_cols.index:
    
    # Integer encode the string categories
    dat = le.fit_transform(data_ohc[col]).astype(np.int)
    
    # Remove the original column from the ohc dataframe
    data_ohc = data_ohc.drop(col, axis=1)

    # One hot encode the data--this returns a sparse array
    new_dat = ohc.fit_transform(dat.reshape(-1,1))

    # Create unique column names
    n_cols = new_dat.shape[1]
    col_names = ['_'.join([col, str(x)]) for x in range(n_cols)]

    # Create the new dataframe
    new_df = pd.DataFrame(new_dat.toarray(), 
                          index=data_ohc.index, 
                          columns=col_names)

    # Append the new data to the dataframe
    data_ohc = pd.concat([data_ohc, new_df], axis=1)
{% endhighlight %}

We used `LabelEncoder` as opposed to `DictVectorizer` because it doesn't require specifying a numerical value for each category anyways.

We could use `DictVectorizer` if we wanted to specify the numbers or only wanted to encode some categories.

{% highlight python %}
#Column differen is as calculated above
data_ohc.shape[1] - data.shape[1]
{% endhighlight %}
> 215

{% highlight python %}
print(data.shape[1])

#Removing the string columns from data df
data = data.drop(num_ohc_cols.index, axis=1)

print(data.shape[1])
{% endhighlight %}
>- 80
- 37

# Training and Test splits
We are using a 70/30 split for train and test respectably. A random state is used to ensure the same split when repeated.

{% highlight python %}
from sklearn.model_selection import train_test_split

y_col = 'SalePrice'

#Split the data that is not one-hot encoded
feature_cols = [x for x in data.columns if x != y_col]
X_data = data[feature_cols]
y_data = data[y_col]

X_train, X_test, y_train, y_test = train_test_split(X_data,y_data,
                                                   test_size=0.3,
                                                   random_state=42)

#Split the data that is one-hot encoded
feature_cols_ohc = [x for x in data_ohc.columns if x != y_col]
X_data_ohc = data_ohc[feature_cols_ohc]
y_data_ohc = data_ohc[y_col]

X_train_ohc, X_test_ohc, y_train_ohc, y_test_ohc = train_test_split(X_data_ohc,
                                                                   y_data_ohc,
                                                                   test_size=0.3,
                                                                   random_state=42)
{% endhighlight %}

{% highlight python %}
#Confirm the indices of both match
(X_train_ohc.index == X_train.index).all()
{% endhighlight %}
>True

For both datasets we will fit a `LinearRegression`.

Then we will calculate the `mean_squared_error`.

{% highlight python %}
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

LR = LinearRegression()

#to store the error value
error_df = list()

#Not encoded dataset
LR = LR.fit(X_train, y_train)
y_train_pred = LR.predict(X_train)
y_test_pred = LR.predict(X_test)

error_df.append(pd.Series({'train': mean_squared_error(y_train, y_train_pred),
                          'test': mean_squared_error(y_test, y_test_pred)},
                         name='no enc'))

#oh-Coded dataset
LR = LR.fit(X_train_ohc, y_train_ohc)
y_train_ohc_pred = LR.predict(X_train_ohc)
y_test_ohc_pred = LR.predict(X_test_ohc)

error_df.append(pd.Series({'train' : mean_squared_error(y_train_ohc,y_train_ohc_pred),
                          'test' : mean_squared_error(y_test_ohc, y_test_ohc_pred)},
                         name='one-hot enc'))
#Assemble the results
error_df = pd.concat(error_df, axis=1)
error_df
{% endhighlight %}

![Output](/assets/img/IntelAI/crossval1.png){:class="img-responsive"}

The error values are quite different. The errors for one-hot encoded data are much higher. This is because the one-hot encoded model is overfitting the model.

# Scaling the datasets
We are going to scale btoh datasets. 

We will be scaling the data using: `StandardScaler`, `MinMaxScaler`,and `MaxAbsScaler`   

{% highlight python %}
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

scalers = {'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'maxabs': MaxAbsScaler()}

training_test_sets = {
    'not_encoded': (X_train, y_train, X_test, y_test),
    'one_hot_encoded': (X_train_ohc, y_train_ohc, X_test_ohc, y_test_ohc)
}

#Find the list of float columns and values to not scale 
# already scaled data

mask = X_train.dtypes == np.float
float_columns = X_train.columns[mask]

# initliase model
LR = LinearRegression()

# Iterate over all possible combinations and get the errors
errors = {}
for encoding_label, (_X_train, _y_train, 
                      _X_test, _y_test) in training_test_sets.items():
    
    for scaler_label, scaler in scalers.items():
        trainingset = _X_train.copy() #copy because we don't want to scale more than once
        testset = _X_test.copy()
        
        trainingset[float_columns] = scaler.fit_transform(trainingset[float_columns])
        testset[float_columns] = scaler.transform(testset[float_columns])
        
        LR.fit(trainingset, y_train)
        predictions = LR.predict(testset)
        
        key = encoding_label + ' - ' + scaler_label + 'scaling'
        errors[key] = mean_squared_error(_y_test, predictions)
        
errors = pd.Series(errors)
print(errors.to_string())
print('-' * 80)
for key, error_val in errors.items():
    print(key,error_val)
{% endhighlight %}
>- not_encoded - standardscaling        1.372182e+09
- not_encoded - minmaxscaling          1.372077e+09
- not_encoded - maxabsscaling          1.372129e+09
- one_hot_encoded - standardscaling    2.574955e+26
- one_hot_encoded - minmaxscaling      8.065328e+09
- one_hot_encoded - maxabsscaling      8.065328e+09

--------------------------------------------------------------------------------
> - not_encoded - standardscaling 1372182358.9345078
- not_encoded - minmaxscaling 1372077190.5328918
- not_encoded - maxabsscaling 1372128637.178544
- one_hot_encoded - standardscaling 2.5749550027093824e+26
- one_hot_encoded - minmaxscaling 8065327607.246822
- one_hot_encoded - maxabsscaling 8065327607.200471

Finally we can plot the Predictions comparing to the actual values of the houses.

{% highlight python %}
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('talk')
sns.set_style('ticks')
sns.set_palette('colorblind')

ax = plt.axes()
#using y_test, y_test_pred
ax.scatter(y_test_ohc, y_test_pred, alpha=0.5)

ax.set(xlabel = 'Ground Truth',
      ylabel = 'Predictions',
       title='Ames, Iowa House Price Predictions vs Truth, using Linear Regression')
{% endhighlight %}

> [Text(0, 0.5, 'Predictions'),
> Text(0.5, 0, 'Ground Truth'),
> Text(0.5, 1.0, 'Ames, Iowa House Price Predictions vs Truth, using Linear Regression')]

![Output](/assets/img/IntelAI/crossval2.png){:class="img-responsive"}


Credits to [Intel AI Academy](https://software.intel.com/en-us/home)

{% highlight python %}

{% endhighlight %}










