---
layout: post
title: "Using ML to filter tasks for RPA"
date: 2020-03-25 12:24:47
image: '/assets/img/'
description: 'Not all process are primed for RPA, and this is not black and white. Some process may have counterparts which are automatable. It is a matter of being able to cherry pick these cases'
tags:
- python
- rpa
- machine learning
- business process
- sklearn
- gridsearchcv
- error metrics
- tasks
- pipeline
categories:
- RPA
twitter_text:
---

# Case Scenario

It's 9 AM and Charlie is just getting into the office. Charlie works in an Insurance firm. One of her main daily tasks is to look through customer interactions and decide what to do with each case.

When a customer calls up the contact centre employee answering the call makes notes on what was discussed on the call. Notes are automatically added to customer's account.

Charlie needs to go read each of these new notes and decide what to do with these cases. These appear as work items on the system. Some of the options might be:

- Send confirmation of insurance cancellation
- Send Policy certificate
- Pass onto Fraud Team
- Pass to Claim Team
- Update Customer's details on the System

Charlie and her team has to process 1000's of notes to go through each day. Most of these notes require a simple action. If an automation was in place to manage the first 4 tasks Charlie and the team can focus on just the updating details.

The first 4 notes rely on reading a `free text` box and unfortunately, there is no logical structure on what the call attendant might write here. It's easy for a human such as Charlie to read and understand what action to take but not rules based enough for a bot.

In this ideal scenario a Machine Learning model can be used as a filter to classify these cases. Once the classification has been done, an RPA solution can start processing these cases. Unfortunately updating of customer details might not be suitable as it will require a high degree of verification and high risk if details include names, address, or bank details.

The task of sending a confirmation of cancellation comes with medium risk and we would want the classification model to have high degree of accuracy. Wrongly issued cancellations will lower customer satisfaction.

The other three tasks such as sending policy certificate, passing the case to different teams comes with low risk as wrongly flagged cases can be salvaged.


# A Code Example

Real company data is not available. But [Taskmaster-1](https://research.google/tools/datasets/taskmaster-1/) contains text descriptions of tasks that are generated from conversations.

this dataset is comprised of conversations between a person and an agent on several different tasks such as ordering food, booking a table at a restaurant, ordering a taxi and so on. Using the conversation text as humans we can judge what this task might be, we should be able to create a model that can predict this as well.

The machine learning models can be made using python or higher level software such as Data Robot. Before using higher level solutions such as Data Robot it's a good idea to understand how much effort it will save.


### Setting up the Data
Import Pandas - to build a dataframe for training

`re` - regular expression, to clean the data later

Load the `.json` file and convert it a pandas dataframe `df`

{% highlight python %}
import pandas as pd
import re
df = pd.read_json (r'/Downloads/self-dialogs.json')
df.keys()
{% endhighlight %}

### The Data has three main parts here:
- `The conversation ID`
- `Instruction-ID` (target label)
- `Utterance` (the actual text/conversation)

We can peek into the text phrases below. Each conversation is separated into blocks of speech in a script fashion.

{% highlight python %}
df["utterances"][0][0].get("text")
{% endhighlight %}

>"Hi, I'm looking to book a table for Korean fod."

### Cleaning up

The data comes as single interactions that make up the whole conversation. To classify this conversation we just need a block text of this conversation and nothing else.

The phrases can now be stitched into single blocks to make it easier to train. Removing the .json tags we don't need and using regex to take away anything that's not alphanumeric.

{% highlight python %}
list_of_utterances = []

for i in range(0,df["utterances"].count()):
    text = ""
    prelim_list =[]
    for j in range(0,len(df["utterances"][i])):
        text=((df["utterances"][i][j].get("text")))
        text = text.lstrip()
        text = text.rstrip()

        text = re.sub('[^A-Za-z0-9]+', ' ', text) #The filter

        prelim_list.append(text+' ') #adding a space before joining sentences
    list_of_utterances.append(''.join(prelim_list))
{% endhighlight %}

### Visualising the text
Now we can see a whole conversation with no special characters. Reading over the text extract you have a good idea what the topic of conversation here is.

The below example is clearly someone ordering a taxi.

{% highlight python %}
print(list_of_utterances[15])
{% endhighlight %}

>I want a pick up today  what is your zip code 90026 ok I can get you an uber or lyft I want an Uber ride How many are being picked up Just two poeple is this for now or a later time This will be after today at 4pm OK where is this begin picked up at I will be at the Hollywood bowl in los angeles Alright wehre do you need drop off I am going to the glendale galleria Ok will there be anything else How long will that ride be and price  It should be 20mins and 18 dollars for that right ok I will take that one sure thing your uber will be ready at 4pm we will let you know when it arives Ok thank you Have a good day

### Readying for training

Adding the labels and text into a dictionary `d` then rewriting the dataframe `df`
The several categories can be seen.

{% highlight python %}
d = {'instruction_id':df["instruction_id"].tolist() , 'utterance': list_of_utterances}
categories = list(set(df["instruction_id"]))
df = pd.DataFrame(data=d)

print(categories)
{% endhighlight %}

>['movie-tickets-1', 'uber-lyft-1', 'restaurant-table-3', 'restaurant-table-1', 'coffee-ordering-1', 'movie-ticket-1', 'auto-repair-appt-1', 'pizza-ordering-2', 'uber-lyft-2', 'movie-tickets-3', 'pizza-ordering-1', 'movie-tickets-2', 'coffee-ordering-2', 'movie-finder', 'restaurant-table-2']


Some of the labels have number suffix's. This is because those conversations have been generated using different methods. For our example we just need to classify the text and don't care about how they were generated so we can take the suffixes out.

{% highlight python %}
df.replace('-\d+', '', regex=True, inplace=True)#consolidating all like for like labels
df.replace('movie-tickets', 'movie-ticket', regex=True, inplace=True)
df['instruction_id'].value_counts()
{% endhighlight %}
>pizza-ordering      1468
>
>coffee-ordering     1376
>
>restaurant-table    1300
>
>movie-ticket        1251
>
>auto-repair-appt    1161
>
>uber-lyft           1098
>
>movie-finder          54
>
>Name: instruction_id, dtype: int64

Dropping `movie-tickets` since it has very low volume. A model with a label that has 54 text samples is not good to train with, especially when we have other samples in the thousands.

{% highlight python %}
df = df[df.instruction_id != 'movie-finder']
df['instruction_id'].value_counts()
{% endhighlight %}

>pizza-ordering      1468
>
>coffee-ordering     1376
>
>restaurant-table    1300
>
>movie-ticket        1251
>
>auto-repair-appt    1161
>
>uber-lyft           1098
>
>Name: instruction_id, dtype: int64

### Data is now clean (-ish)

We'll be using scikit-learn packages for processing the data.
FIrstly, split into test and train sets. 77% train and 33% test.

{% highlight python %}

{% endhighlight %}

![Output](/assets/img/IntelAI/SVM1.png){:class="img-responsive"}

First we combine the both white and red wine data.

{% highlight python %}
import pandas as pd
import numpy as np

data_red = pd.read_csv("winequality-red.csv",sep=';')
data_red['color'] = 1 #redwine

print(data_red.shape)

data_white = pd.read_csv("winequality-white.csv",sep=';')
data_white['color'] = 0 #whitewine

print(data_white.shape)

data = data_red.merge(data_white, how='outer')
fields = list(data.columns)
print(fields)
{% endhighlight %}

> - (1599, 13)
- (4898, 13)
- ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality', 'color']

Now, lets look at a pairplot to spot any obvious separations.

{% highlight python %}
import matplotlib.pyplot as plt
import seaborn as sns

g = sns.pairplot(data, diag_kind='hist',hue='color')
{% endhighlight %}

![Output](/assets/img/IntelAI/SVM2.png){:class="img-responsive"}

There doesn't seem to be any obvious clusters of wine. Let's now drop off the non-chemical features.

{% highlight python %}
data = data.drop(columns=['color','quality'])
fields = list(data.columns)
print(fields)
{% endhighlight %}

> ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

Since we have high-dimensional data(more than 3), Lets see if we can drop some less useful features using using PCA.

{% highlight python %}
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
%matplotlib inline

X = data[fields]
X = scale(X)

pca = PCA(n_components=11)

pca.fit(X)

var= pca.explained_variance_ratio_

var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print(var1)
plt.plot(var1)

pca = PCA(n_components=9)
pca.fit(X)
X1=pca.fit_transform(X)

#print(X1)
{% endhighlight %}
> [27.54 50.21 64.36 73.18 79.72 85.24 90.   94.56 97.62 99.69 99.99]

![Output](/assets/img/IntelAI/SVM3.png){:class="img-responsive"}

The least important feature component still accounts for 28% variance and will not be taken out.

K-means clustering used to cluster the data points and upto 200 clusters were generated iteratevely. We are using silhouette scores (internal validation) to check for compactness and well-separatedness of these clusters.

The average scores for each iterations are then plotted.

{% highlight python %}
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import itertools

#X = data[fields]

plotx = []
ploty = []

plot_dict = {}

for n_clusters in range (2, 50):

    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(X1)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X1, cluster_labels)
    plotx.append(n_clusters)
    ploty.append(silhouette_avg)

    plot_dict[n_clusters] = silhouette_avg
#     print("For n_clusters =", n_clusters,
#           "The average silhouette_score is :", silhouette_avg)

plt.plot(plotx,ploty)
plt.xlabel('no. of clusters')
plt.ylabel('avg. silhouette score')
print("number of clusters with the highest avg. silhoutte score is : "+str(max(plot_dict.keys(), key=lambda k: plot_dict[k])))
{% endhighlight %}

> number of clusters with the highest avg. silhoutte score is : 2
![Output](/assets/img/IntelAI/SVM1.png){:class="img-responsive"}

# What predicts wine quality

A **Pearson correlation** was used to identify which features correlate with wine quality. It looks as if **higher the alcohol content** the **higher the quality**. Lower density and volatile acidity also correlated with better quality as seen in the pairwise correlation chart in Figure 2. Only the top 5 correlated features were carried over to the SVM models.

The figure below shows Pearson Pairwise correlation of features to wine quality.

![Output](/assets/img/IntelAI/SVM4.png){:class="img-responsive"}

The study by [Cortez et al., 2009] used Support Vector Machines and thus it will be used again to predict the target class, quality. A significant delay was noticed when training SVM's without normalized data.

After scaling, test and train sets were created(70/30 split). GridSearchCV was used to tune the hyper-parameters of the SVM model. The model's evaluation is shown below. Two cases are shown, model trained and tested with all the data(model A), and second case omitting qualities with less than 1000 samples(model B) i.e qualities of 5,6,7. Model B achieved better scores than Model A, but can still be improved with more data and expertise.

![Output](/assets/img/IntelAI/SVM5.png){:class="img-responsive"}

Shown above is the Confusion matrix heat-map of the two different SVM models.

![Output](/assets/img/IntelAI/SVM6.png){:class="img-responsive"}

Firstly loading and combing both wine data.

{% highlight python %}
import pandas as pd
import numpy as np

data_red = pd.read_csv("winequality-red.csv",sep=';')
data_red['color'] = 1 #redwine

print(data_red.shape)

data_white = pd.read_csv("winequality-white.csv",sep=';')
data_white['color'] = 0 #whitewine

print(data_white.shape)

data = data_red.merge(data_white, how='outer')


######SUPPRESS_SAMPLES_WITH_VERY_LOW_COUNT####
data = data.drop(data[data.quality == 9].index)
data = data.drop(data[data.quality == 3].index)
data = data.drop(data[data.quality == 8].index)
data = data.drop(data[data.quality == 4].index)
#data = data.drop(data[data.quality == 7].index)
{% endhighlight %}

>- (1599, 13)
- (4898, 13)

There is an imbalance in the datasets, with more than double the samples for white wines. As seen below, there is very low samples for wine qualities of `4,8,3,9`, comparing with `6,5,7`.

{% highlight python %}
data.quality.value_counts()
{% endhighlight %}

>- 6    2836
- 5    2138
- 7    1079
- Name: quality, dtype: int64

Separating all the features as `X`, and the target quality as `y`.

{% highlight python %}
fields = list(data.columns[:-2])
fields.append('color')#adding color back
X = data[fields]
y = data['quality']
print(fields)
{% endhighlight %}

> ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'color']

Perfoming pairwise correlation with quality to see which features are highly correlated.

{% highlight python %}
correlations = data[fields].corrwith(y)
correlations.sort_values(inplace=True)

fields = correlations.map(abs).sort_values().iloc[-5:].index
print(fields) #prints the top two abs correlations
{% endhighlight %}

> Index(['color', 'chlorides', 'volatile acidity', 'density', 'alcohol'], dtype='object')

{% highlight python %}
import matplotlib.pyplot as plt
import seaborn as sns

ax = correlations.plot(kind='bar')
ax.set(ylim=[-1, 1], ylabel='pearson correlation')
{% endhighlight %}

![Output](/assets/img/IntelAI/SVM4.png){:class="img-responsive"}

Looks like `alcohol` and `density` are the most correlated with **`quality`**. As per [Cortez et al., 2009], let's use a SVM to classify wine quality.

# SVM
SVM converges faster when features are scaled. If the model is senstive to magnitudes its generally a good idea to scale so one feature doesn't get more influence than the other(in terms of scale).

{% highlight python %}
from sklearn.preprocessing import MinMaxScaler

X = data[fields]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=['%s_scaled' % fld for fld in fields])
print(X.columns) #scaled columns
{% endhighlight %}

> Index(['color_scaled', 'chlorides_scaled', 'volatile acidity_scaled',
       'density_scaled', 'alcohol_scaled'],
      dtype='object')

Splitting into test and train sets.

{% highlight python %}
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
{% endhighlight %}

GridSearchCV to tune hyperparameters for the SVM.

{% highlight python %}
parameters = {'kernel':('linear', 'rbf'), 'C':[.1, 1, 10], 'gamma':[.5, 1, 2, 10]}

SVC_Gaussian = svm.SVC(gamma='scale')
gscv = GridSearchCV(SVC_Gaussian, param_grid=parameters, cv=5)
gscv.fit(X_train, y_train)
{% endhighlight %}
> GridSearchCV(cv=5, error_score='raise-deprecating',
       estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False),
       fit_params=None, iid='warn', n_jobs=None,
       param_grid={'kernel': ('linear', 'rbf'), 'C': [0.1, 1, 10], 'gamma': [0.5, 1, 2, 10]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring=None, verbose=0)

Printing the best parameters.

{% highlight python %}
print(gscv.best_estimator_)
print(gscv.best_params_)
{% endhighlight %}

> SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=10, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
{'C': 1, 'gamma': 10, 'kernel': 'rbf'}

Now we train the SVM model using the above parameters.

{% highlight python %}
SVC_Gaussian = svm.SVC(kernel='rbf', gamma=10, C=10)
SVC_Gaussian.fit(X_train, y_train)
{% endhighlight %}

> SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=10, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)


{% highlight python %}
y_pred = SVC_Gaussian.predict(X_test)
{% endhighlight %}

{% highlight python %}
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

metrics = list()
cm = dict()


# Preciision, recall, f-score from the multi-class support function
precision, recall, fscore, _ = score(y_test, y_pred, average='weighted')

# The usual way to calculate accuracy
accuracy = accuracy_score(y_test, y_pred)


metrics.append(pd.Series({'precision':precision, 'recall':recall,
                          'fscore':fscore, 'accuracy':accuracy},
                         name='Model'))

metrics = pd.concat(metrics, axis=1)

metrics
{% endhighlight %}
![Output](/assets/img/IntelAI/SVM7.png){:class="img-responsive"}

{% highlight python %}
y_pred = SVC_Gaussian.predict(X_test)
{% endhighlight %}

{% highlight python %}
y_pred
#y_test
{% endhighlight %}
> array([7, 5, 6, ..., 5, 6, 5])

{% highlight python %}
import matplotlib.pyplot as plt


# Last, the confusion matrix
cm = confusion_matrix(y_test, y_pred)

axList = axList.flatten()
fig.set_size_inches(15, 15)

axList[-1].axis('off')

plot = sns.heatmap(cm, annot=True, fmt='d');

plt.tight_layout()
{% endhighlight %}

![Output](/assets/img/IntelAI/SVM8.png){:class="img-responsive"}

Credits to [Intel AI Academy](https://software.intel.com/en-us/home)

{% highlight python %}

{% endhighlight %}
