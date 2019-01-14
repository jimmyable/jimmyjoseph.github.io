---
layout: post
title: "Regularization and Gradient Descent"
date: 2018-11-27 07:26:47
image: '/assets/img/'
description: 'Regularization or normalization changes the scaling for highly varied data. Stochastic Gradient Descent is also explored.'
tags:
- python
- regularization
- machine learning
- numpy
- scikit
- stochastic gradient descent
- ridge regression
- scaling
- lasso regression
- elastic net
categories:
- Intel AI Machine Leaning
twitter_text: 
---

# Regularization and Gradient Descent

In this worked example we will explore `regression`, `polynomial features`, and `regularization` using very simple sparse data. 

First we import the data, which contains and x and y columns of noisy data.

{% highlight python %}
import pandas as pd, numpy as np

data = pd.read_csv("data/X_Y_Sinusoid_Data.csv")
{% endhighlight %}

Now we will generate 100 equally spaced **x** data points and over the range of 0 to 1. Using these points we will generate the **y** points of *ground truth* from the equation $y = sin(2\pi x)$

{% highlight python %}
X_real = np.linspace(0, 1, 100)
Y_real = np.sin(2*np.pi*X_real)
{% endhighlight %}

Let's see how this looks plotted out.

{% highlight python %}
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')
sns.set_context('talk')
sns.set_palette('inferno')

#plotting the noisy input data
ax = data.set_index('x')['y'].plot(ls='',marker='o', label='data')
#plotting the ground truth
ax.plot(X_real, Y_real, ls='--', marker='', label='ground truth')

ax.legend()
ax.set(xlabel='x', ylabel='y')
{% endhighlight %}

![Output](/assets/img/IntelAI/SGD1.png){:class="img-responsive"}

# Polynomial Features

First we use `PolynomialFeatures` to create 20th order polynomial features. 20 because we have 20 data points currently.

{% highlight python %}
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

degree = 20
pf = PolynomialFeatures(degree)
lr = LinearRegression()
{% endhighlight %}

Now we will fit the data using `LinearRegression`

{% highlight python %}
X_data = data[['x']]
Y_data = data['y']

#fitting the model
X_poly = pf.fit_transform(X_data)
lr = lr.fit(X_poly, Y_data)
Y_pred_lr = lr.predict(X_poly)

{% endhighlight %}

Let's plot the results.

{% highlight python %}
plt.plot(X_data, Y_data, marker='o', ls='', label='data', alpha=1)
plt.plot(X_real, Y_real, ls='--',label='ground truth')
plt.plot(X_data, Y_pred_lr, marker='^', alpha=0.5, label='polynomial prediction')
plt.legend()
ax = plt.gca()
ax.set(xlabel='x data', ylabel='y data')
{% endhighlight %}

![Output](/assets/img/IntelAI/SGD2.png){:class="img-responsive"}

# Regularization
Using the polynomial features data we can perform the regression. `RidgeRegression`(L2) using $\alpha$=0.001, and `LassoRegression`(L1) using $\alpha$=0.0001

{% highlight python %}
from sklearn.linear_model import Ridge, Lasso

#ridge regression model
rr = Ridge(alpha=0.001)
rr = rr.fit(X_poly, Y_data)
Y_pred_rr = rr.predict(X_poly)

#lasso regression model
lassor = Lasso(alpha=0.0001)
lassor = lassor.fit(X_poly, Y_data)
Y_pred_lassor = lassor.predict(X_poly)
{% endhighlight %}

Plotting the predicted results again.

{% highlight python %}
plt.plot(X_data, Y_data, marker='o', ls='', label='data')
plt.plot(X_real, Y_real, ls='--', label='ground truth')
plt.plot(X_data, Y_pred_lr, label='linear regression', marker='^',alpha=.5)
plt.plot(X_data, Y_pred_rr, label='ridge regression', marker='^', alpha=.5)
plt.plot(X_data, Y_pred_lassor, label='lasso regression', marker='^', alpha=.5)

plt.legend()

ax = plt.gca()
ax.set(xlabel='x',ylabel='y')
{% endhighlight %}

![Output](/assets/img/IntelAI/SGD3.png){:class="img-responsive"}

The lasso and ridge regression seems to be much better fit. Lets see how the magnitude of their coefficients compare to the liner regression.

{% highlight python %}
coeffs = pd.DataFrame()
coeffs['linear_regression'] = lr.coef_.ravel()
coeffs['ridge_regression'] = rr.coef_.ravel()
coeffs['lasso_regression'] = lassor.coef_.ravel()
coeffs = coeffs.applymap(abs) #makes all values absolutes

coeffs.describe()
{% endhighlight %}

![Output](/assets/img/IntelAI/SGD4.png){:class="img-responsive"}

PLotting the maginitude of coefficients. The Linear Regression will be plotted on a different y-axis since its magnitude is very large.

{% highlight python %}
colors = sns.color_palette()

#setting up both y-axes
ax1 = plt.axes()
ax2 = ax1.twinx()

#linear regression data
ax1.plot(lr.coef_.ravel(), color=colors[0], marker='o',
         label='linear regression')

#ridge and lasso
ax2.plot(rr.coef_.ravel(), color=colors[3], marker='o',
         label='ridge regression')
ax2.plot(lassor.coef_.ravel(), color=colors[5], marker='o',
        label='lasso regression')

#axes scales
ax1.set_ylim(-2e14, 2e14)
ax2.set_ylim(-25,25)

#combine the legends
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2)

#plot axes labels
ax1.set(xlabel='coefficients',ylabel='linear regression')
ax2.set(ylabel='ridge and lasso regression')
{% endhighlight %}

![Output](/assets/img/IntelAI/SGD5.png){:class="img-responsive"}

The large magnitudes of the data caused overfitting. Regularization penalised the bigger weights by modifying the cost function. More about this [here](https://www.commonlounge.com/discussion/e3561f62d9c848c4b936f8d5abbdd1b3).

# Test-Train split and Skewing
This example uses the Ames Housing prices data.

Firstly, importing the data. Then removing all null values, and one-hot encoding categoricals.

{% highlight python %}
data = pd.read_csv('Intel-ML101-Class4/data/Ames_Housing_Sales.csv')

#get a pd.series of all string categs
one_hot_encode_cols = data.dtypes[data.dtypes == np.object] #filtering string categs
one_hot_encode_cols = one_hot_encode_cols.index.tolist() #list of categ fields

#encoding these columns as categories means o-h-c will work even when the data is split
for col in one_hot_encode_cols:
    data[col] = pd.Categorical(data[col])

#one hot encoding
data = pd.get_dummies(data, columns=one_hot_encode_cols)
{% endhighlight %}

Splitting the data into test and train data sets.

{% highlight python %}
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.3, random_state=42)
{% endhighlight %}

Some columns have skewed features, to which we will apply log transformations. The label `SalePrice` also contains some. but we will be ignoring it for now.

{% highlight python %}
#list of float columns to check for skewing
mask = data.dtypes == np.float
float_cols = data.columns[mask]

skew_limit = 0.75
skew_vals = train[float_cols].skew()

skew_cols = (skew_vals
            .sort_values(ascending=False)
            .to_frame()
            .rename(columns={0:'Skew'})
            .query('abs(Skew) > {0}'.format(skew_limit)))

skew_cols
{% endhighlight %}

![Output](/assets/img/IntelAI/SGD6.png){:class="img-responsive"}

Transform all the columns with skew > 0.75, apart from `SalePrice`.

Let's visualise at what happens to one of these features after we apply `np.log1p`.

{% highlight python %}
feature = "BsmtFinSF1"
fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))
train[feature].hist(ax=ax_before)
train[feature].apply(np.log1p).hist(ax=ax_after)

ax_before.set(title='before np.log1p', ylabel='frequency', xlabel='value')
ax_after.set(title='after np.log1p', ylabel='frequency', xlabel='value')

fig.suptitle('Field "{}"'.format(feature));

{% endhighlight %}

![Output](/assets/img/IntelAI/SGD7.png){:class="img-responsive"}

Applying to all the features.

{% highlight python %}
for col in skew_cols.index.tolist():
    if col == "SalePrice":
        continue
    train[col] = np.log1p(train[col])
    test[col] = test[col].apply(np.log1p)
{% endhighlight %}

Separating features from predictor.

{% highlight python %}
feature_cols = [x for x in train.columns if x != 'SalePrice']
X_train = train[feature_cols]
y_train = train['SalePrice']

X_test = test[feature_cols]
y_test = test['SalePrice']
{% endhighlight %}

# Linear Regression and RMSE

We will create a function `rmse` which calculates the **root mean squared error** using the sklearn's `mean_squared_error` method.

{% highlight python %}
from sklearn.metrics import mean_squared_error

def rmse (ytrue, ypredicted):
    return np.sqrt(mean_squared_error(ytrue, ypredicted))
{% endhighlight %}

Fitting a basic linear regression model.

{% highlight python %}
from sklearn.linear_model import LinearRegression

linearRegression = LinearRegression().fit(X_train, y_train)
{% endhighlight %}

Printing the root mean-squared error for this model.

{% highlight python %}
linearRegression_rmse = rmse(y_test, linearRegression.predict(X_test))
print(linearRegression_rmse)
{% endhighlight %}

>306369.68342316244

Plotting the predicted and actual sale price based on this model.

{% highlight python %}
f = plt.figure(figsize=(6,6))
ax = plt.axes()

ax.plot(y_test, linearRegression.predict(X_test),
        marker='o', ls='', ms=4.0)

lim = (0, y_test.max())

ax.set(xlabel='Actual Price',
      ylabel='Predicted Price',
      ylim=lim,
      xlim=lim,
      title='Linear Regression Results')
{% endhighlight %}

![Output](/assets/img/IntelAI/SGD8.png){:class="img-responsive"}

# Rigde Regression
Rigde Regression uses L2 normalization to reduce the magnitude of the coefficients. This can be helpful when there is high variance in the data. The regularization functions in sklearn has cross-validation built in.

We are going to fit a (non-cross validated) Ridge model to a range of $\alpha$ values and plot the RMSE using cross validated error function created above.

Alpha values:
    $$[0.005, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 80]$$

Now for the `RidgeCV` method, it's not possible to get the alpha values for the models that weren't selected. The resulting error values and $\alpha$ values are very similair to what is obtained above.

Finally we can compare the error values of prior and the Ridge models.

{% highlight python %}
from sklearn.linear_model import RidgeCV

alphas = [0.005, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 80]

ridgeCV = RidgeCV(alphas=alphas,
                 cv=4).fit(X_train, y_train)

ridgeCV_rmse = rmse(y_test, ridgeCV.predict(X_test))

print(ridgeCV.alpha_, ridgeCV_rmse)
{% endhighlight %}
>15.0 32169.176205672433

# Lasso Regression

Just like `RidgeCV`, `LassoCV` is a function but uses L1 regularization. L1 regularization will selectively shrink some coefficients, which is effectively performing feature elimination.

`LassoCV` does not have allow a scoring function but the `rmse` function created above can be used for model evaluation.

*There is also `ElasticNetCV` which uses a combination of both L1 and L2 normalisation/regularisation*

Using alphas:
$$[1e-5, 5e-5, 0.0001, 0.0005]$$

Fit a Lasso model using cross validation and determine the optimum value for $\alpha$, and the RMSE using the function above. The magnitude of alphas can be different to the Ridge model.

{% highlight python %}
from sklearn.linear_model import LassoCV

alphas2 = np.array([1e-5, 5e-5, 0.0001, 0.0005])

lassoCV = LassoCV(alphas=alphas2, 
                 max_iter=5e4,
                 cv=3).fit(X_train, y_train)

lassoCV_rmse = rmse(y_test, lassoCV.predict(X_test))

print(lassoCV.alpha_, lassoCV_rmse) #Lasso is slower
{% endhighlight %}

>0.0005 39257.393991448225

We can determine how many of these features remain non-zero.

{% highlight python %}
print('Of {} coefficients, {} are non-zero with Lasso.'.format(len(lassoCV.coef_),
                                                              len(lassoCV.coef_.nonzero()[0])))
{% endhighlight %}
>Of 294 coefficients, 274 are non-zero with Lasso.

# ElasticNet

Using the same alphas as Lasso, we will now try an `ElasticNetCV`, with L1 ratios between 0.1 and 0.9

{% highlight python %}
from sklearn.linear_model import ElasticNetCV

l1_ratios = np.linspace(0.1, 0.9, 9)

elasticNetCV = ElasticNetCV(alphas=alphas2,
                           l1_ratio=l1_ratios,
                           max_iter=1e4).fit(X_train, y_train)

elasticNetCV_rmse = rmse(y_test, elasticNetCV.predict(X_test))

print(elasticNetCV.alpha_, elasticNetCV.l1_ratio_, elasticNetCV_rmse)
{% endhighlight %}

>0.0005 0.1 35001.23429607454

Its easiest to compare the RMSE calculations for all models in a table.

{% highlight python %}
rmse_vals = [linearRegression_rmse, ridgeCV_rmse, lassoCV_rmse, elasticNetCV_rmse]

labels = ['Linear', 'Ridge', 'Lasso','ElasticNet']

rmse_df = pd.Series(rmse_vals, index=labels).to_frame()
rmse_df.rename(columns={0:'RMSE'}, inplace=1)
rmse_df
{% endhighlight %}

![Output](/assets/img/IntelAI/SGD9.png){:class="img-responsive"}

Plotting the actual and predicted housing prices as before.

{% highlight python %}
f = plt.figure(figsize=(6,6))
ax = plt.axes()

labels = ['Ridge', 'Lasso', 'ElasticNet']

models = [ridgeCV, lassoCV, elasticNetCV]

for mod, lab in zip(models, labels):
    ax.plot(y_test, mod.predict(X_test),
           marker='o', ls='', ms=3.0, label=lab)
    
leg = plt.legend(frameon=True)
leg.get_frame().set_edgecolor('black')
leg.get_frame().set_linewidth(1.0)

ax.set(xlabel='Actual Price',
      ylabel='Predicted Price',
      title='Linear Regression Results')
{% endhighlight %}

![Output](/assets/img/IntelAI/SGD10.png){:class="img-responsive"}

# Stochastic Gradient Descent

Linear Models in general are sensitive to scaling. SGD is **very senstive** to scaling

And a high value of learning rate can cause the algorithmn to diverge, while too low of a value may take too long to converge.

Fitting a stochastic gradient descent model without a regularization penalty(the relavant parameter is `penalty`)

{% highlight python %}

# Import SGDRegressor and prepare the parameters

from sklearn.linear_model import SGDRegressor

model_parameters_dict = {
    'Linear': {'penalty': 'none'},
    'Lasso': {'penalty': 'l2',
           'alpha': lassoCV.alpha_},
    'Ridge': {'penalty': 'l1',
           'alpha': ridgeCV_rmse},
    'ElasticNet': {'penalty': 'elasticnet', 
                   'alpha': elasticNetCV.alpha_,
                   'l1_ratio': elasticNetCV.l1_ratio_}
}

new_rmses = {}
for modellabel, parameters in model_parameters_dict.items():
    # following notation passes the dict items as arguments
    SGD = SGDRegressor(**parameters)
    SGD.fit(X_train, y_train)
    new_rmses[modellabel] = rmse(y_test, SGD.predict(X_test))

rmse_df['RMSE-SGD'] = pd.Series(new_rmses)
rmse_df
{% endhighlight %}

![Output](/assets/img/IntelAI/SGD11.png){:class="img-responsive"}

We can see that the error values are vert high. This means the algorithm is divering, and can be due to high scaling/learning rate.

What happens if we adjust the learning rate?

{% highlight python %}
# Import SGDRegressor and prepare the parameters

from sklearn.linear_model import SGDRegressor

model_parameters_dict = {
    'Linear': {'penalty': 'none'},
    'Lasso': {'penalty': 'l2',
           'alpha': lassoCV.alpha_},
    'Ridge': {'penalty': 'l1',
           'alpha': ridgeCV_rmse},
    'ElasticNet': {'penalty': 'elasticnet', 
                   'alpha': elasticNetCV.alpha_,
                   'l1_ratio': elasticNetCV.l1_ratio_}
}

new_rmses = {}
for modellabel, parameters in model_parameters_dict.items():
    # following notation passes the dict items as arguments
    SGD = SGDRegressor(eta0=1e-7,**parameters)
    SGD.fit(X_train, y_train)
    new_rmses[modellabel] = rmse(y_test, SGD.predict(X_test))

rmse_df['RMSE-SGD-learningrate'] = pd.Series(new_rmses)
rmse_df
{% endhighlight %}

![Output](/assets/img/IntelAI/SGD12.png){:class="img-responsive"}

The error values are much lower. Now let's try scaling the training data.

{% highlight python %}
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

{% endhighlight %}

{% highlight python %}
# Import SGDRegressor and prepare the parameters

from sklearn.linear_model import SGDRegressor

model_parameters_dict = {
    'Linear': {'penalty': 'none'},
    'Lasso': {'penalty': 'l2',
           'alpha': lassoCV.alpha_},
    'Ridge': {'penalty': 'l1',
           'alpha': ridgeCV_rmse},
    'ElasticNet': {'penalty': 'elasticnet', 
                   'alpha': elasticNetCV.alpha_,
                   'l1_ratio': elasticNetCV.l1_ratio_}
}

new_rmses = {}
for modellabel, parameters in model_parameters_dict.items():
    # following notation passes the dict items as arguments
    SGD = SGDRegressor(**parameters)
    SGD.fit(X_train_scaled, y_train)
    new_rmses[modellabel] = rmse(y_test, SGD.predict(X_test_scaled))

rmse_df['RMSE-SGD-learningrate-Scaled'] = pd.Series(new_rmses)
rmse_df
{% endhighlight %}

![Output](/assets/img/IntelAI/SGD13.png){:class="img-responsive"}

Credits to [Intel AI Academy](https://software.intel.com/en-us/home)

{% highlight python %}

{% endhighlight %}










