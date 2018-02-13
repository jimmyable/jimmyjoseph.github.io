---
layout: post
title: "Chocolate Visualisations"
date: 2018-01-27 22:55:47
image: '/assets/img/'
description: 'Using some simple data visualisation techniques on a chocolate dataset'
tags:
- python
- scatterplot
- barplot
- numpy
- pandas
- matplotlib
- dataviz
categories:
- ML & Data Mining 
twitter_text: 
---

## Intro
We are going to use a Chocolate dataset collected from [Kaggle](https://www.kaggle.com/rtatman/chocolate-bar-ratings/data), to perform some simple Data visualisation techniques.
Numpy and Pandas library must be installed. Code and data can be retrieved at [DataMiningSnippets](https://github.com/jimmyjoseph1295/DataMiningSnippets)

**Numpy** - A library for Python, it is used for scientific computing

**Pandas** - This library allows for creation of data structures within Python and can perform manipulations and analysis on the data

Installed using `pip install pandas numpy`

## Initialisation
The first step is to initialise the data into a Pandas dataframe.
So we import `os` to do os operations such as finding current filepath and opening files.
`Numpy` and `pandas` modules are also imported.

Then the `.csv` file is converted into a pandas dataframe.

{% highlight python %}
import os, numpy, pandas as pd

filepath = os.sep.join(['data/flavors_of_cacao.csv'])
print(filepath)
data = pd.read_csv(filepath)
data.head()
{% endhighlight %}

![Output](/assets/img/DataMiningSnippets/1.png){:class="img-responsive"}

## Built-in functions
We can now easily inspect the data using some of the built-in pandas commands, such as the number of rows, column names and data types within the dataframe.

{% highlight python %}
# Number of rows
print("no. of rows = ",data.shape[0],"\n")

# Column names
print("the columns are",data.columns.tolist(),"\n")

# Data types
print(data.dtypes)
{% endhighlight %}

We can drop the columns "REF" and "Review Date" because we can. Also, its also not super useful anyways. 

{% highlight python %}
data.drop(columns=['REF','Review\nDate'])
{% endhighlight %}

Next let's count up the bean types. 
We can also do a mean, median, and quartiles for Cocoa Percentages just by using `.describe()`

{% highlight python %}
beantypes = data['Bean\nType'].value_counts()
print(beantypes,"\n")

data['Cocoa\nPercent'] = data['Cocoa\nPercent'].str.replace('%', '')
cocoa_percent_int=pd.to_numeric(data['Cocoa\nPercent'],downcast='integer')
cocoa_percent_int.describe()
{% endhighlight %}

![Output](/assets/img/DataMiningSnippets/7.png){:class="img-responsive"}

## Plotting

Now we can make a scatterplot of the Rating given against Cocoa Strength. We need to import `matplotlib` to achieve this.
{% highlight python %}
import matplotlib.pyplot as plt
%matplotlib inline

# A simple scatter plot with Matplotlib
ax = plt.axes()

ax.scatter(cocoa_percent_int, data['Rating'])

# Label the axes
ax.set(xlabel='Cocoa Strength',
       ylabel='Rating',
       title='Cocoa Strength vs Rating')
{% endhighlight %}

![Output](/assets/img/DataMiningSnippets/2.png){:class="img-responsive"}

We can use pandas' `.value_counts()` to calculate the frequencies for example how many times a Company is repeated, this will give us how many different products they make.

Using `matplotlib` again now we can make a bar chart, this time lets plot the top 6 Companies

{% highlight python %}
bc = plt.axes()

Companyfreq=data['Company\xa0\n(Maker-if known)'].value_counts()
x=[] #init empty lists
y=[]
for i in range (0,5):
    x.append(Companyfreq.axes[0][i])
    y.append(Companyfreq[i])
    
bc.bar(x,y)

{% endhighlight %}

![Output](/assets/img/DataMiningSnippets/3.png){:class="img-responsive"}

## Seaborn

But we haven't used numpy yet? What's the point in even making trivial plots that can be made on excel? Well, below is a randomly generated data plotted using `Seaborn`, it makes plots not easily do-able in excel.

{% highlight python %}
from scipy.stats import kendalltau
import seaborn as sns
sns.set(style="ticks")

rs = np.random.RandomState(11)
x = rs.gamma(2, size=1000)
y = -.5 * x + rs.normal(size=1000)

sns.jointplot(x, y, kind="hex", stat_func=kendalltau, color="#4CB391")
{% endhighlight %}

![Output](/assets/img/DataMiningSnippets/5.png){:class="img-responsive"}

But we can't just end with random data plots, so let's make one more plot. Since we've introduced Seaborn, this time we will use it again and plot the counts of each ratings.
{% highlight python %}
sns.countplot(x = 'Rating', data=data) #"Rating" is picked and plotted
{% endhighlight %}

![Output](/assets/img/DataMiningSnippets/6.png){:class="img-responsive"}









