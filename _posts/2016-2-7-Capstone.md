---
layout: post
title: Classification using Logistic Regression on Loans
---

Introduction:
This project analyzes a dataset in an attempt to determine what causes the good employees to leave their current jobs. The first step is to clean the data (Part I) and visualize it (Part II) using various types of graphs and charts to help establish clear relationships between various features. Part III analyzes the dataset using various Machine Learning algorithms and picks the best algorithm to model this dataset based on various metrics. Lastly, a conclusion on the entire analysis is provided.
Part I
Data Cleaning
This initial stage of the project is to clean the data in such a way that all the important attributes of the dataset be retained and unhindered for analysis purposes. Hence, the initial cleaning phase of the project imports the dataset and the libraries needed for the cleaning stages of the project. A check for Null Values/Missing Values is conducted (none are present). The multiple categorical variables in the dataset are created into dummy variables which is necessary for later stages of the analysis. These new variables are afterwards added to the original dataset and the initial variables from which dummification was done are removed to avoid redundancy. Lastly, the dataset is normalized due to the need to apply machine learning algorithms to it in later stages of the project.
In [1]:
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

%matplotlib inline
In [2]:
HR = pd.read_csv('HR_comma_sep.csv')
HR.head(10)
Out[2]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary
0	0.38	0.53	2	157	3	0	1	0	sales	low
1	0.80	0.86	5	262	6	0	1	0	sales	medium
2	0.11	0.88	7	272	4	0	1	0	sales	medium
3	0.72	0.87	5	223	5	0	1	0	sales	low
4	0.37	0.52	2	159	3	0	1	0	sales	low
5	0.41	0.50	2	153	3	0	1	0	sales	low
6	0.10	0.77	6	247	4	0	1	0	sales	low
7	0.92	0.85	5	259	5	0	1	0	sales	low
8	0.89	1.00	5	224	5	0	1	0	sales	low
9	0.42	0.53	2	142	3	0	1	0	sales	low
In [3]:
HR[HR['satisfaction_level'] == np.nan]
HR[HR['last_evaluation'] == np.nan]
HR[HR['number_project'] == np.nan]
HR[HR['average_montly_hours'] == np.nan]
HR[HR['time_spend_company'] == np.nan]
HR[HR['Work_accident'] == np.nan]
HR[HR['left'] == np.nan]
HR[HR['sales'] == np.nan]
HR[HR['salary'] == np.nan]
Out[3]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary
In [4]:
# Create a heatmap, to check for null values.

plt.figure(figsize=(10,7))
sns.heatmap(HR.isnull(),yticklabels=False, cbar=False, cmap='viridis')
Out[4]:
<matplotlib.axes._subplots.AxesSubplot at 0x114c6ee50>

In [5]:
HR['sales'].value_counts()
Out[5]:
sales          4140
technical      2720
support        2229
IT             1227
product_mng     902
marketing       858
RandD           787
accounting      767
hr              739
management      630
Name: sales, dtype: int64
In [6]:
HR['salary'].value_counts()
Out[6]:
low       7316
medium    6446
high      1237
Name: salary, dtype: int64
In [7]:
HR['left'].value_counts()
Out[7]:
0    11428
1     3571
Name: left, dtype: int64
In [8]:
# Create dummy variables for all the departments

sales = pd.get_dummies(HR.sales)

sales.columns = ["IT", "RandD", "accounting", "hr", "management", "marketing", "product_mng", "sales_department", "support", "technical"]

sales.head()
Out[8]:
IT	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical
0	0	0	0	0	0	0	0	1	0	0
1	0	0	0	0	0	0	0	1	0	0
2	0	0	0	0	0	0	0	1	0	0
3	0	0	0	0	0	0	0	1	0	0
4	0	0	0	0	0	0	0	1	0	0
In [9]:
# Create dummy variables for the salary

salary = pd.get_dummies(HR.salary)

salary.columns = ['low', 'medium', 'high']

salary.head()
Out[9]:
low	medium	high
0	0	1	0
1	0	0	1
2	0	0	1
3	0	1	0
4	0	1	0
In [10]:
# Create dummy variable for stayed/left

left = pd.get_dummies(HR.left)

left.columns = ['left', 'stayed']

left.head(10)
Out[10]:
left	stayed
0	0	1
1	0	1
2	0	1
3	0	1
4	0	1
5	0	1
6	0	1
7	0	1
8	0	1
9	0	1
In [11]:
# Concatenate the original dataset with the newly created salary and sales dummified variables

HR = pd.concat([HR, salary, sales], axis = 1)

HR.head(10)
Out[11]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary	...	IT	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical
0	0.38	0.53	2	157	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
1	0.80	0.86	5	262	6	0	1	0	sales	medium	...	0	0	0	0	0	0	0	1	0	0
2	0.11	0.88	7	272	4	0	1	0	sales	medium	...	0	0	0	0	0	0	0	1	0	0
3	0.72	0.87	5	223	5	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
4	0.37	0.52	2	159	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
5	0.41	0.50	2	153	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
6	0.10	0.77	6	247	4	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
7	0.92	0.85	5	259	5	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
8	0.89	1.00	5	224	5	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
9	0.42	0.53	2	142	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
10 rows × 23 columns
In [12]:
HR.dtypes
Out[12]:
satisfaction_level       float64
last_evaluation          float64
number_project             int64
average_montly_hours       int64
time_spend_company         int64
Work_accident              int64
left                       int64
promotion_last_5years      int64
sales                     object
salary                    object
low                        uint8
medium                     uint8
high                       uint8
IT                         uint8
RandD                      uint8
accounting                 uint8
hr                         uint8
management                 uint8
marketing                  uint8
product_mng                uint8
sales_department           uint8
support                    uint8
technical                  uint8
dtype: object
In [13]:
# Create a new column 

salary_map = {'low': 1, 'medium': 2, 'high': 3}

HR['salary_variable'] = HR['salary'].apply(lambda x: salary_map[x])

HR.head()
Out[13]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary	...	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical	salary_variable
0	0.38	0.53	2	157	3	0	1	0	sales	low	...	0	0	0	0	0	0	1	0	0	1
1	0.80	0.86	5	262	6	0	1	0	sales	medium	...	0	0	0	0	0	0	1	0	0	2
2	0.11	0.88	7	272	4	0	1	0	sales	medium	...	0	0	0	0	0	0	1	0	0	2
3	0.72	0.87	5	223	5	0	1	0	sales	low	...	0	0	0	0	0	0	1	0	0	1
4	0.37	0.52	2	159	3	0	1	0	sales	low	...	0	0	0	0	0	0	1	0	0	1
5 rows × 24 columns
In [14]:
del HR['salary']

del HR['sales']
In [15]:
HR.head()
Out[15]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	low	medium	...	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical	salary_variable
0	0.38	0.53	2	157	3	0	1	0	0	1	...	0	0	0	0	0	0	1	0	0	1
1	0.80	0.86	5	262	6	0	1	0	0	0	...	0	0	0	0	0	0	1	0	0	2
2	0.11	0.88	7	272	4	0	1	0	0	0	...	0	0	0	0	0	0	1	0	0	2
3	0.72	0.87	5	223	5	0	1	0	0	1	...	0	0	0	0	0	0	1	0	0	1
4	0.37	0.52	2	159	3	0	1	0	0	1	...	0	0	0	0	0	0	1	0	0	1
5 rows × 22 columns
In [16]:
HR.dtypes
Out[16]:
satisfaction_level       float64
last_evaluation          float64
number_project             int64
average_montly_hours       int64
time_spend_company         int64
Work_accident              int64
left                       int64
promotion_last_5years      int64
low                        uint8
medium                     uint8
high                       uint8
IT                         uint8
RandD                      uint8
accounting                 uint8
hr                         uint8
management                 uint8
marketing                  uint8
product_mng                uint8
sales_department           uint8
support                    uint8
technical                  uint8
salary_variable            int64
dtype: object
In [17]:
from sklearn.preprocessing import MinMaxScaler
X_scaled = MinMaxScaler().fit_transform(HR.astype(float))
X_scaled = pd.DataFrame(X_scaled, columns = HR.columns)

# Min Max Scaler

X_scaled.head(10)
Out[17]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	low	medium	...	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical	salary_variable
0	0.318681	0.265625	0.0	0.285047	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
1	0.780220	0.781250	0.6	0.775701	0.500	0.0	1.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.5
2	0.021978	0.812500	1.0	0.822430	0.250	0.0	1.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.5
3	0.692308	0.796875	0.6	0.593458	0.375	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
4	0.307692	0.250000	0.0	0.294393	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
5	0.351648	0.218750	0.0	0.266355	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
6	0.010989	0.640625	0.8	0.705607	0.250	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
7	0.912088	0.765625	0.6	0.761682	0.375	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
8	0.879121	1.000000	0.6	0.598131	0.375	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
9	0.362637	0.265625	0.0	0.214953	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
10 rows × 22 columns
Part II
Exploratory Data Analysis (EDA)
In this section, we try to visualize the data on employees using various plots, graphs, charts in order to get an idea of how different features in the dataset relate to each other.
The below represents a heatmap of correlations of features. Some interesting observations from this heatmap: Quiet intuitively, you can see that the relationship between satisfaction level and people that left the company is negative. It's also interesting to note that there is a positive correlation between the number of hours employees work and their evaluation. These and other relationships will be explored more deeply in the rest of the EDA.
In [18]:
correlation = X_scaled.corr()
plt.figure(figsize=(20,20))
sns.heatmap(correlation, vmax=10, square=True,annot=True,cmap=None, linewidths=1)

plt.title('Correlation between features')
Out[18]:
<matplotlib.text.Text at 0x118f3a050>

In [19]:
# Exported csv file to work with in Tableau
X_scaled.to_csv("Updated_Capstone.csv")
In [20]:
X_scaled['left'].value_counts()
Out[20]:
0.0    11428
1.0     3571
Name: left, dtype: int64
In [21]:
stayed = np.where(X_scaled['left'] == 0)[0]
left = np.where(X_scaled['left'] == 1)[0]
For a given level of satisfaction, there is definitely an intuitive pattern that the higher someone's salary is, the less they are likely to leave. This relationship is a lot more apparent in the distinction between the high income earners vs low & median income earners as a group. The difference in the people that left between low and median income salaries is not significant. Hence, most of the people that leave are not receiving a high salary, which means that you if you would like to retain these employees, you should increase their pay!
In [22]:
# plot of Satisfaction vs Last Evaluation
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.satisfaction_level[stayed], X_scaled.salary_variable[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.satisfaction_level[left], X_scaled.salary_variable[left], alpha=0.1, color='r')
ax.set_title("satisfaction_level vs salary")
ax.set_xlabel("satisfaction_level")
ax.set_ylabel("salary")
plt.show()

In the Satisfaction vs Last Evaluation plot below, we can note some interesting observations. Firstly, there is a sizeable cluster of people with strong evaluations and low satisfaction levels leaving. Secondly, there is a significant group of people leaving that are slighly below average in performance and evaluation. Lastly, and by far the most interesting observation of the three, there is a decent size of a not very dense cluster of people leaving that have done extremely well on their evaluations (0.7 to 1 on a 0-1 scale) and and have very high satisfaction rates, albeit not perfect (0.7-0.9 on a 0-1 scale). That particular group of employees also have many people that stay, but a significant proportion of that group is leaving. So the most surprising thing to observe is that people can be satisfied with the job and still leave! There are several possibilites for why this might be possible. People might enjoy their job but think they can do even better elsewhere, whether its income or career growth (or a combination of the two, considering that those things tend to go hand in hand). Another possibility is that people might not be honest in surveys because of fear of being reprimanded for negative surveys.
In [23]:
# plot of Satisfaction vs Last Evaluation
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.satisfaction_level[stayed], X_scaled.last_evaluation[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.satisfaction_level[left], X_scaled.last_evaluation[left], alpha=0.1, color='r')
ax.set_title("satisfaction_level vs last_evaluation")
ax.set_xlabel("satisfaction_level")
ax.set_ylabel("last_evaluation")
plt.show()

The Satisfaction vs Promotion during the last five years plot below shows a very strong relationship in several interesting ways! People who have been promoted don't leave very often! However, such promotions are not common enough! Therefore, if you don't want good employees to leave, promoting them will really help!
In [24]:
# plot of Satisfaction vs Promotion during last five years
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.satisfaction_level[stayed], X_scaled.promotion_last_5years[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.satisfaction_level[left], X_scaled.promotion_last_5years[left], alpha=0.1, color='r')
ax.set_title("satisfaction_level vs promotion_last_5years")
ax.set_xlabel("satisfaction_level")
ax.set_ylabel("promotion_last_5years")
plt.show()

The below scatter plot compares the average working hours against the time spent at the company and how it impacts the employee turnover rate. There are multiple interesting relationships to note. For employees that just started out, the majority tend to stick around for some time even if they work substantial hours. However, for employees who have been at the company for a while, there is a significant tendency to leave the company if working substantial hours. Consequently, we can observe that very few employees stick around at the company for a long time period because the ones who were working long hours almost all left! However, those that do stick around for a long time (relatively few) tend to all be devoted and not leave
There are several conjectures for why this is possible. One is that employees have worked for a significant amount of hours and put in a lot of time and effort expecting a quick promotion and/or pay increase (refer to graph above for relationship between promotions and staying/leaving) that they have not received and leave for elsewhere. Another reason might be that people have tried working for a little bit of time, realized that they are being overworked and have started looking for other opportunities to seek employment.
In [25]:
# plot of Average_monthly_hours vs time_spend_company
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.average_montly_hours[stayed], X_scaled.time_spend_company[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.average_montly_hours[left], X_scaled.time_spend_company[left], alpha=0.1, color='r')
ax.set_title("average_monthly_hours vs time_spend_company")
ax.set_xlabel("average_monthly_hours")
ax.set_ylabel("time_spend_company")
plt.show()

In [26]:
# Imported from Tableau
from IPython.display import Image
Image(filename = 'Average Monthly Hours vs Satisfaction Level.png', width = 1000)
Out[26]:

There does not appear to be a strong relationship between accidents and people leaving the company, hence, people stay/leave regardless of the accidents they might have experienced.
In [27]:
# plot of work_accident vs time_spend_company
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.Work_accident[stayed], X_scaled.time_spend_company[stayed], alpha=0.5, color='g')
ax.scatter(X_scaled.Work_accident[left], X_scaled.time_spend_company[left], alpha=0.5, color='r')
ax.set_title("Work_accident vs time_spend_company")
ax.set_xlabel("Work_accident")
ax.set_ylabel("time_spend_company")
plt.show()

The below stripplot compares number of projects done by an employee versus the number of promotions they have received over the past 5 years. It is interesting to note that the number of projects completed has very insignificant impact on whether a person leaves. On the contrary, the promotions received during the past 5 years is of critical importance. The majority of people who have received a promotion stay. Almost all those who were not promoted leave.
The data seems insufficient to be able to determine why the number of projects is not related to promotions. There are at least several possibilities. Perhaps the number of projects refers to the number assigned (vs completed) or that the projects are not all of equal difficulty and not equally time consuming leading to distorted hypothesis that the number of projects might necessarily correspond to an employee's productivity.
In [28]:
import seaborn as sns

sns.stripplot("number_project", "promotion_last_5years", data=X_scaled, hue="left", jitter = True)
Out[28]:
<matplotlib.axes._subplots.AxesSubplot at 0x117949d90>

The Satisfaction level distribution below shows that the majority of people are at least moderately satisfied with their job. Therefore, for the people that are satisfied and do well, efforts should be placed on making them even more happy by promoting them. For the people that do poorly or mediocre, either studies should be placed on determining why they are not doing well and trying to improve their performance or attempting to replace them with better employees (this would depend on a cost/benefit analysis) done by the company regarding this issue.
In [29]:
# Histogram distribution of Satisfaction Level

f,ax1 = plt.subplots(1,1)
sns.distplot(X_scaled['satisfaction_level'],bins=10,norm_hist=False)
plt.show()

The distribution (which is a decent approximation of a normal distribution with almost no tails and some kurtosis present) of the last evaluation graph shows us there is a sizeable portion of people that do well in their evaluations, hence the company should try to do as much as possible to keep those people!
In [30]:
# Histogram distribution of Last Evaluation


f,ax1 = plt.subplots(1,1)
sns.distplot(X_scaled['last_evaluation'],bins=10,norm_hist=False)
plt.show()

The below plot shows the ratios of people who stay in the company by department and the ratio of people who leave the company by department. It is clear from the chart that the departments with the most number of employees also have the highest ratios of both people who have left and those that stayed at the company. This tells us there should be particular effort place on retaining good employees that work in those departments, in particular the sales, technical and support departments, respectively. The largest efforts should be placed on retaining the good employees in these groups because they have the most people leaving.
In [34]:
# This plot is to compare the people that left across different departments
# The non-department variables are deleted to only keep the departments

del X_scaled['satisfaction_level']
del X_scaled['last_evaluation']
del X_scaled['number_project']
del X_scaled['average_montly_hours']
del X_scaled['time_spend_company']
del X_scaled['Work_accident']
del X_scaled['promotion_last_5years']
del X_scaled['low']
del X_scaled['medium']
del X_scaled['high']
del X_scaled['salary_variable']


left=X_scaled.groupby('left').mean()
left=left.transpose()
plt.figure(figsize=(20,10))
left.plot(kind='bar')
Out[34]:
<matplotlib.axes._subplots.AxesSubplot at 0x11a7bcd10>
<matplotlib.figure.Figure at 0x11a360850>

Part III
Predictive Modelling
In this section below, multiple models will be run that are applicable to classification. Afterwards, various scoring metrics will be calculated for the models and a determination will be made as to which model is based for this dataset.
In [174]:
# read in the data & create matrices
X = X_scaled[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", \
"time_spend_company", "Work_accident", "promotion_last_5years", "salary_variable",\
"technical", "support", "IT", "product_mng", "marketing", "RandD", "accounting",\
"hr", "management", "sales_department"]]
y = X_scaled["left"]
In [175]:
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
Logistic Regression Model
In [33]:
# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
model = LogisticRegression()

# fit the model with data
mdl = model.fit(X_train, y_train)

# predict the response for new observations

logit = model.predict(X_test)

# sklearn output to check that ran on right data split.

len(logit)
Out[33]:
3750
In [34]:
# These represent the y-intercept and coefficients for all the variables in the logistic regression.

print(model.intercept_)
print(model.coef_)
[ 0.57305398]
[[-3.71487625  0.45484967 -1.50619896  0.8704181   1.97590531 -1.45425383
-1.26024249 -1.32512177  0.22588571  0.23497924 -0.01282827  0.07509077
0.13925062 -0.47542127  0.26606552  0.3920348  -0.44451555  0.17251241]]
In [35]:
# Prediction Accuracy for Logistic Regression

from sklearn import metrics
print(metrics.accuracy_score(y_test, logit))
0.796
In [36]:
from sklearn.cross_validation import cross_val_score
In [37]:
# 10-fold cross-validation for Logistic Regression to find the CV score
mdl = model.fit(X_train, y_train)
scores = cross_val_score(mdl, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.794205514996
In [38]:
from sklearn import metrics
confusion = metrics.confusion_matrix(y_test, logit)
print ((confusion))
[[2651  211]
[ 554  334]]
In [39]:
y_pred_prob = model.predict_proba(X_test)[:, 1]
In [40]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob, bins=5)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[40]:
<matplotlib.text.Text at 0x1181a6a90>

In [41]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [42]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, logit)


plot_confusion_matrix(cm, title='LogisticRegression: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [43]:
# Roc_curve for Logistic Regression Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, logit)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Logistic Regression Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

K Nearest Nearbours Model
K=1
In [44]:
from sklearn.neighbors import KNeighborsClassifier
In [45]:
# K-Nearest Neighbours with K=1
knn = KNeighborsClassifier(n_neighbors=1, metric = 'euclidean')
In [46]:
y = knn.fit(X_train, y_train)

KNN = knn.predict(X_test)

# sklearn output to check that ran on right data split.

len(KNN)
Out[46]:
3750
In [47]:
# Cross fold validation for K=1

md2 = knn.fit(X_train, y_train)
scores = cross_val_score(md2, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.962043001151
In [48]:
# Prediction Accuracy.

from sklearn import metrics
print(metrics.accuracy_score(y_test, KNN))
0.967466666667
In [49]:
confusion = metrics.confusion_matrix(y_test, KNN)
print confusion
[[2775   87]
[  35  853]]
In [50]:
# Classification Accuracy confirmation between manual calculation and formula for K=1

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, KNN))
0.967466666667
0.967466666667
In [51]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=1
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN))
0.0325333333333
0.0325333333333
In [52]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for K=1

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, KNN))
0.960585585586
0.960585585586
In [53]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for K=1


print(TN / float(TN + FP))
0.969601677149
In [54]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for K=1

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, KNN))
0.907446808511
0.907446808511
In [55]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=1
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN))
0.0325333333333
0.0325333333333
In [56]:
y_pred_prob_knn = knn.predict_proba(X_test)[:, 1]
In [57]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_knn, bins=5)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[57]:
<matplotlib.text.Text at 0x1181d6b90>

In [58]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [59]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, KNN)


plot_confusion_matrix(cm, title='KNN for K=1: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [60]:
# Roc_curve for KNN=1 Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, KNN)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('KNN=1 Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

K=5
In [61]:
# K-Nearest Neighbours with K=5

knn = KNeighborsClassifier(n_neighbors=5, metric = 'euclidean')
y = knn.fit(X_train, y_train)
KNN5 = knn.predict(X_test)

# sklearn output to check that ran on right data split.

len(KNN5)
Out[61]:
3750
In [62]:
# Cross Validation for K=5 Cross-Fold Validation

md2 = knn.fit(X_train, y_train)
scores = cross_val_score(md2, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.938309179837
In [63]:
# Prediction Accuracy on KNN=5 for accuracy score

from sklearn import metrics
print(metrics.accuracy_score(y_test, KNN5))
0.941333333333
In [64]:
confusion = metrics.confusion_matrix(y_test, KNN5)
print confusion
[[2731  131]
[  89  799]]
In [65]:
# Classification Accuracy confirmation between manual calculation and formula for K=5

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, KNN5))
0.941333333333
0.941333333333
In [66]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=5
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN5))
0.0586666666667
0.0586666666667
In [67]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for K=5

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, KNN5))
0.899774774775
0.899774774775
In [68]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for K=5

print(TN / float(TN + FP))
0.954227812718
In [69]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for K=5

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, KNN5))
0.859139784946
0.859139784946
In [70]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=5
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN5))
0.0586666666667
0.0586666666667
In [71]:
# Roc_curve for KNN=5 Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, KNN5)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [72]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [73]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, KNN5)


plot_confusion_matrix(cm, title='KNN for N=5: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [74]:
y_pred_prob_knn = knn.predict_proba(X_test)[:, 1]
In [75]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_knn, bins=5)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[75]:
<matplotlib.text.Text at 0x11acdea10>

In [ ]:

Grid Search for optimal N value in KNN
In [101]:
from sklearn.grid_search import GridSearchCV
/Users/Misha/anaconda/lib/python2.7/site-packages/sklearn/grid_search.py:43: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.
DeprecationWarning)
In [102]:
k_potentials = list(range(1, 200))
print(k_potentials)
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]
In [103]:
# Set Nearest Neighbours equal to k
knn = KNeighborsClassifier(n_neighbors=k_potentials, metric = 'euclidean')
# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_potentials)
# instantiate the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
# fit the grid with data
grid.fit(X_train, y_train)
# view the results
grid.grid_scores_
Out[103]:
[mean: 0.96240, std: 0.00591, params: {'n_neighbors': 1},
mean: 0.95884, std: 0.00543, params: {'n_neighbors': 2},
mean: 0.94568, std: 0.00463, params: {'n_neighbors': 3},
mean: 0.94622, std: 0.00555, params: {'n_neighbors': 4},
mean: 0.93875, std: 0.00620, params: {'n_neighbors': 5},
mean: 0.94284, std: 0.00673, params: {'n_neighbors': 6},
mean: 0.93795, std: 0.00746, params: {'n_neighbors': 7},
mean: 0.93839, std: 0.00819, params: {'n_neighbors': 8},
mean: 0.93351, std: 0.00932, params: {'n_neighbors': 9},
mean: 0.93493, std: 0.00848, params: {'n_neighbors': 10},
mean: 0.93048, std: 0.00825, params: {'n_neighbors': 11},
mean: 0.93253, std: 0.00810, params: {'n_neighbors': 12},
mean: 0.92933, std: 0.00898, params: {'n_neighbors': 13},
mean: 0.92968, std: 0.00824, params: {'n_neighbors': 14},
mean: 0.92719, std: 0.00747, params: {'n_neighbors': 15},
mean: 0.92817, std: 0.00830, params: {'n_neighbors': 16},
mean: 0.92559, std: 0.00873, params: {'n_neighbors': 17},
mean: 0.92728, std: 0.00821, params: {'n_neighbors': 18},
mean: 0.92497, std: 0.00821, params: {'n_neighbors': 19},
mean: 0.92470, std: 0.00830, params: {'n_neighbors': 20},
mean: 0.92204, std: 0.00804, params: {'n_neighbors': 21},
mean: 0.92186, std: 0.00867, params: {'n_neighbors': 22},
mean: 0.91902, std: 0.00856, params: {'n_neighbors': 23},
mean: 0.91777, std: 0.00850, params: {'n_neighbors': 24},
mean: 0.91564, std: 0.00809, params: {'n_neighbors': 25},
mean: 0.91457, std: 0.00913, params: {'n_neighbors': 26},
mean: 0.91244, std: 0.00927, params: {'n_neighbors': 27},
mean: 0.91244, std: 0.00873, params: {'n_neighbors': 28},
mean: 0.91101, std: 0.00878, params: {'n_neighbors': 29},
mean: 0.91110, std: 0.00853, params: {'n_neighbors': 30},
mean: 0.90906, std: 0.00855, params: {'n_neighbors': 31},
mean: 0.90941, std: 0.00906, params: {'n_neighbors': 32},
mean: 0.90773, std: 0.00922, params: {'n_neighbors': 33},
mean: 0.90728, std: 0.00991, params: {'n_neighbors': 34},
mean: 0.90550, std: 0.00946, params: {'n_neighbors': 35},
mean: 0.90417, std: 0.00891, params: {'n_neighbors': 36},
mean: 0.90292, std: 0.00887, params: {'n_neighbors': 37},
mean: 0.90097, std: 0.00881, params: {'n_neighbors': 38},
mean: 0.90035, std: 0.00948, params: {'n_neighbors': 39},
mean: 0.89875, std: 0.01009, params: {'n_neighbors': 40},
mean: 0.89759, std: 0.01009, params: {'n_neighbors': 41},
mean: 0.89626, std: 0.00955, params: {'n_neighbors': 42},
mean: 0.89537, std: 0.01016, params: {'n_neighbors': 43},
mean: 0.89270, std: 0.01221, params: {'n_neighbors': 44},
mean: 0.89172, std: 0.01244, params: {'n_neighbors': 45},
mean: 0.88950, std: 0.01089, params: {'n_neighbors': 46},
mean: 0.88906, std: 0.01150, params: {'n_neighbors': 47},
mean: 0.88666, std: 0.01137, params: {'n_neighbors': 48},
mean: 0.88577, std: 0.01143, params: {'n_neighbors': 49},
mean: 0.88523, std: 0.01132, params: {'n_neighbors': 50},
mean: 0.88452, std: 0.01149, params: {'n_neighbors': 51},
mean: 0.88372, std: 0.01237, params: {'n_neighbors': 52},
mean: 0.88283, std: 0.01227, params: {'n_neighbors': 53},
mean: 0.88141, std: 0.01166, params: {'n_neighbors': 54},
mean: 0.88114, std: 0.01085, params: {'n_neighbors': 55},
mean: 0.88061, std: 0.01041, params: {'n_neighbors': 56},
mean: 0.87990, std: 0.01105, params: {'n_neighbors': 57},
mean: 0.87910, std: 0.00974, params: {'n_neighbors': 58},
mean: 0.87892, std: 0.00997, params: {'n_neighbors': 59},
mean: 0.87803, std: 0.01108, params: {'n_neighbors': 60},
mean: 0.87768, std: 0.01119, params: {'n_neighbors': 61},
mean: 0.87688, std: 0.01152, params: {'n_neighbors': 62},
mean: 0.87652, std: 0.01182, params: {'n_neighbors': 63},
mean: 0.87590, std: 0.01129, params: {'n_neighbors': 64},
mean: 0.87554, std: 0.01148, params: {'n_neighbors': 65},
mean: 0.87474, std: 0.01183, params: {'n_neighbors': 66},
mean: 0.87412, std: 0.01223, params: {'n_neighbors': 67},
mean: 0.87394, std: 0.01196, params: {'n_neighbors': 68},
mean: 0.87341, std: 0.01210, params: {'n_neighbors': 69},
mean: 0.87314, std: 0.01231, params: {'n_neighbors': 70},
mean: 0.87270, std: 0.01219, params: {'n_neighbors': 71},
mean: 0.87226, std: 0.01168, params: {'n_neighbors': 72},
mean: 0.87154, std: 0.01185, params: {'n_neighbors': 73},
mean: 0.87074, std: 0.01104, params: {'n_neighbors': 74},
mean: 0.87057, std: 0.01132, params: {'n_neighbors': 75},
mean: 0.87003, std: 0.01129, params: {'n_neighbors': 76},
mean: 0.86968, std: 0.01148, params: {'n_neighbors': 77},
mean: 0.86754, std: 0.01056, params: {'n_neighbors': 78},
mean: 0.86692, std: 0.01041, params: {'n_neighbors': 79},
mean: 0.86568, std: 0.01028, params: {'n_neighbors': 80},
mean: 0.86594, std: 0.01024, params: {'n_neighbors': 81},
mean: 0.86461, std: 0.00937, params: {'n_neighbors': 82},
mean: 0.86390, std: 0.00915, params: {'n_neighbors': 83},
mean: 0.86239, std: 0.00956, params: {'n_neighbors': 84},
mean: 0.86257, std: 0.01001, params: {'n_neighbors': 85},
mean: 0.86088, std: 0.01023, params: {'n_neighbors': 86},
mean: 0.86088, std: 0.01051, params: {'n_neighbors': 87},
mean: 0.85981, std: 0.01214, params: {'n_neighbors': 88},
mean: 0.85937, std: 0.01208, params: {'n_neighbors': 89},
mean: 0.85830, std: 0.01087, params: {'n_neighbors': 90},
mean: 0.85821, std: 0.01108, params: {'n_neighbors': 91},
mean: 0.85616, std: 0.00995, params: {'n_neighbors': 92},
mean: 0.85652, std: 0.00954, params: {'n_neighbors': 93},
mean: 0.85519, std: 0.00892, params: {'n_neighbors': 94},
mean: 0.85545, std: 0.00957, params: {'n_neighbors': 95},
mean: 0.85368, std: 0.00900, params: {'n_neighbors': 96},
mean: 0.85332, std: 0.00845, params: {'n_neighbors': 97},
mean: 0.85199, std: 0.00923, params: {'n_neighbors': 98},
mean: 0.85190, std: 0.00926, params: {'n_neighbors': 99},
mean: 0.85021, std: 0.00825, params: {'n_neighbors': 100},
mean: 0.85039, std: 0.00930, params: {'n_neighbors': 101},
mean: 0.84834, std: 0.01015, params: {'n_neighbors': 102},
mean: 0.84825, std: 0.01016, params: {'n_neighbors': 103},
mean: 0.84559, std: 0.00954, params: {'n_neighbors': 104},
mean: 0.84559, std: 0.00922, params: {'n_neighbors': 105},
mean: 0.84416, std: 0.00977, params: {'n_neighbors': 106},
mean: 0.84452, std: 0.00978, params: {'n_neighbors': 107},
mean: 0.84372, std: 0.00935, params: {'n_neighbors': 108},
mean: 0.84399, std: 0.00922, params: {'n_neighbors': 109},
mean: 0.84256, std: 0.00943, params: {'n_neighbors': 110},
mean: 0.84239, std: 0.00930, params: {'n_neighbors': 111},
mean: 0.84132, std: 0.00941, params: {'n_neighbors': 112},
mean: 0.84087, std: 0.00924, params: {'n_neighbors': 113},
mean: 0.84034, std: 0.00864, params: {'n_neighbors': 114},
mean: 0.84016, std: 0.00875, params: {'n_neighbors': 115},
mean: 0.83981, std: 0.00849, params: {'n_neighbors': 116},
mean: 0.83972, std: 0.00862, params: {'n_neighbors': 117},
mean: 0.83865, std: 0.00779, params: {'n_neighbors': 118},
mean: 0.83892, std: 0.00782, params: {'n_neighbors': 119},
mean: 0.83883, std: 0.00798, params: {'n_neighbors': 120},
mean: 0.83892, std: 0.00810, params: {'n_neighbors': 121},
mean: 0.83661, std: 0.00577, params: {'n_neighbors': 122},
mean: 0.83705, std: 0.00582, params: {'n_neighbors': 123},
mean: 0.83670, std: 0.00582, params: {'n_neighbors': 124},
mean: 0.83714, std: 0.00545, params: {'n_neighbors': 125},
mean: 0.83732, std: 0.00565, params: {'n_neighbors': 126},
mean: 0.83741, std: 0.00610, params: {'n_neighbors': 127},
mean: 0.83732, std: 0.00647, params: {'n_neighbors': 128},
mean: 0.83741, std: 0.00634, params: {'n_neighbors': 129},
mean: 0.83759, std: 0.00626, params: {'n_neighbors': 130},
mean: 0.83750, std: 0.00609, params: {'n_neighbors': 131},
mean: 0.83759, std: 0.00639, params: {'n_neighbors': 132},
mean: 0.83767, std: 0.00616, params: {'n_neighbors': 133},
mean: 0.83732, std: 0.00640, params: {'n_neighbors': 134},
mean: 0.83776, std: 0.00641, params: {'n_neighbors': 135},
mean: 0.83723, std: 0.00690, params: {'n_neighbors': 136},
mean: 0.83794, std: 0.00675, params: {'n_neighbors': 137},
mean: 0.83687, std: 0.00710, params: {'n_neighbors': 138},
mean: 0.83687, std: 0.00718, params: {'n_neighbors': 139},
mean: 0.83607, std: 0.00744, params: {'n_neighbors': 140},
mean: 0.83625, std: 0.00725, params: {'n_neighbors': 141},
mean: 0.83501, std: 0.00739, params: {'n_neighbors': 142},
mean: 0.83536, std: 0.00761, params: {'n_neighbors': 143},
mean: 0.83430, std: 0.00774, params: {'n_neighbors': 144},
mean: 0.83536, std: 0.00776, params: {'n_neighbors': 145},
mean: 0.83510, std: 0.00746, params: {'n_neighbors': 146},
mean: 0.83527, std: 0.00732, params: {'n_neighbors': 147},
mean: 0.83412, std: 0.00716, params: {'n_neighbors': 148},
mean: 0.83492, std: 0.00704, params: {'n_neighbors': 149},
mean: 0.83474, std: 0.00740, params: {'n_neighbors': 150},
mean: 0.83483, std: 0.00759, params: {'n_neighbors': 151},
mean: 0.83492, std: 0.00751, params: {'n_neighbors': 152},
mean: 0.83536, std: 0.00788, params: {'n_neighbors': 153},
mean: 0.83492, std: 0.00833, params: {'n_neighbors': 154},
mean: 0.83492, std: 0.00841, params: {'n_neighbors': 155},
mean: 0.83492, std: 0.00815, params: {'n_neighbors': 156},
mean: 0.83510, std: 0.00822, params: {'n_neighbors': 157},
mean: 0.83527, std: 0.00826, params: {'n_neighbors': 158},
mean: 0.83554, std: 0.00832, params: {'n_neighbors': 159},
mean: 0.83554, std: 0.00778, params: {'n_neighbors': 160},
mean: 0.83519, std: 0.00745, params: {'n_neighbors': 161},
mean: 0.83519, std: 0.00757, params: {'n_neighbors': 162},
mean: 0.83536, std: 0.00715, params: {'n_neighbors': 163},
mean: 0.83527, std: 0.00767, params: {'n_neighbors': 164},
mean: 0.83536, std: 0.00778, params: {'n_neighbors': 165},
mean: 0.83492, std: 0.00759, params: {'n_neighbors': 166},
mean: 0.83510, std: 0.00790, params: {'n_neighbors': 167},
mean: 0.83527, std: 0.00820, params: {'n_neighbors': 168},
mean: 0.83492, std: 0.00775, params: {'n_neighbors': 169},
mean: 0.83456, std: 0.00746, params: {'n_neighbors': 170},
mean: 0.83447, std: 0.00782, params: {'n_neighbors': 171},
mean: 0.83376, std: 0.00770, params: {'n_neighbors': 172},
mean: 0.83376, std: 0.00731, params: {'n_neighbors': 173},
mean: 0.83270, std: 0.00737, params: {'n_neighbors': 174},
mean: 0.83287, std: 0.00736, params: {'n_neighbors': 175},
mean: 0.83261, std: 0.00701, params: {'n_neighbors': 176},
mean: 0.83234, std: 0.00676, params: {'n_neighbors': 177},
mean: 0.83225, std: 0.00719, params: {'n_neighbors': 178},
mean: 0.83207, std: 0.00718, params: {'n_neighbors': 179},
mean: 0.83181, std: 0.00746, params: {'n_neighbors': 180},
mean: 0.83190, std: 0.00741, params: {'n_neighbors': 181},
mean: 0.83243, std: 0.00808, params: {'n_neighbors': 182},
mean: 0.83261, std: 0.00827, params: {'n_neighbors': 183},
mean: 0.83225, std: 0.00819, params: {'n_neighbors': 184},
mean: 0.83279, std: 0.00779, params: {'n_neighbors': 185},
mean: 0.83118, std: 0.00688, params: {'n_neighbors': 186},
mean: 0.83154, std: 0.00684, params: {'n_neighbors': 187},
mean: 0.82950, std: 0.00716, params: {'n_neighbors': 188},
mean: 0.83003, std: 0.00689, params: {'n_neighbors': 189},
mean: 0.82870, std: 0.00649, params: {'n_neighbors': 190},
mean: 0.82878, std: 0.00704, params: {'n_neighbors': 191},
mean: 0.82665, std: 0.00624, params: {'n_neighbors': 192},
mean: 0.82718, std: 0.00654, params: {'n_neighbors': 193},
mean: 0.82478, std: 0.00895, params: {'n_neighbors': 194},
mean: 0.82550, std: 0.00846, params: {'n_neighbors': 195},
mean: 0.82230, std: 0.00974, params: {'n_neighbors': 196},
mean: 0.82274, std: 0.01041, params: {'n_neighbors': 197},
mean: 0.81936, std: 0.00926, params: {'n_neighbors': 198},
mean: 0.81972, std: 0.00963, params: {'n_neighbors': 199}]
In [104]:
# create a list of the mean scores only
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print(grid_mean_scores)
[0.9623966574806649, 0.9588407858476309, 0.9456840608054049, 0.94621744155036, 0.9387501111209885, 0.9428393634989777, 0.9379500400035559, 0.9383945239576851, 0.9335052004622633, 0.9349275491154769, 0.9304827095741843, 0.9325273357631789, 0.9293270512934483, 0.9296826384567517, 0.9271935283136279, 0.9281713930127122, 0.9255933860787625, 0.9272824251044537, 0.9249711085429816, 0.9247044181705041, 0.9220375144457285, 0.9218597208640769, 0.9190150235576495, 0.9177704684860877, 0.9156369455062672, 0.914570184016357, 0.9124366610365365, 0.9124366610365365, 0.911014312383323, 0.9111032091741488, 0.9090585829851542, 0.9094141701484576, 0.9077251311227664, 0.9072806471686372, 0.9055027113521202, 0.9041692594897324, 0.9029247044181705, 0.9009689750200017, 0.9003466974842208, 0.8987465552493555, 0.8975908969686194, 0.8962574451062316, 0.8953684771979732, 0.8927015734731976, 0.8917237087741132, 0.8895012890034669, 0.8890568050493377, 0.8866565916970397, 0.8857676237887813, 0.8852342430438261, 0.8845230687172193, 0.8837229975997867, 0.8828340296915281, 0.8814116810383145, 0.881144990665837, 0.8806116099208818, 0.879900435594275, 0.8791003644768424, 0.8789225708951907, 0.8780336029869322, 0.8776780158236288, 0.8768779447061961, 0.8765223575428926, 0.8759000800071117, 0.8755444928438083, 0.8747444217263757, 0.8741221441905948, 0.873944350608943, 0.8734109698639879, 0.8731442794915104, 0.8726997955373811, 0.8722553115832519, 0.871544137256645, 0.8707440661392124, 0.8705662725575607, 0.8700328918126056, 0.8696773046493022, 0.8675437816694818, 0.8669215041337007, 0.8656769490621389, 0.8659436394346164, 0.8646101875722286, 0.8638990132456218, 0.8623877678015823, 0.862565561383234, 0.8608765223575429, 0.8608765223575429, 0.8598097608676327, 0.8593652769135034, 0.8582985154235933, 0.8582096186327673, 0.8561649924437728, 0.8565205796070762, 0.8551871277446884, 0.8554538181171659, 0.853675882300649, 0.8533202951373455, 0.8519868432749578, 0.851897946484132, 0.8502089074584408, 0.8503867010400925, 0.8483420748510979, 0.848253178060272, 0.8455862743354965, 0.8455862743354965, 0.8441639256822828, 0.8445195128455862, 0.8437194417281536, 0.8439861321006311, 0.8425637834474176, 0.8423859898657658, 0.8413192283758556, 0.8408747444217264, 0.8403413636767713, 0.8401635700951195, 0.8398079829318161, 0.8397190861409903, 0.8386523246510801, 0.8389190150235577, 0.8388301182327318, 0.8389190150235577, 0.8366076984620855, 0.8370521824162148, 0.8366965952529114, 0.8371410792070406, 0.8373188727886923, 0.8374077695795181, 0.8373188727886923, 0.8374077695795181, 0.8375855631611698, 0.837496666370344, 0.8375855631611698, 0.8376744599519957, 0.8373188727886923, 0.8377633567428215, 0.8372299759978665, 0.8379411503244732, 0.8368743888345631, 0.8368743888345631, 0.8360743177171304, 0.8362521112987821, 0.8350075562272202, 0.8353631433905236, 0.8342963819006134, 0.8353631433905236, 0.835096453018046, 0.8352742465996977, 0.8341185883189617, 0.8349186594363943, 0.8347408658547426, 0.8348297626455685, 0.8349186594363943, 0.8353631433905236, 0.8349186594363943, 0.8349186594363943, 0.8349186594363943, 0.835096453018046, 0.8352742465996977, 0.8355409369721754, 0.8355409369721754, 0.8351853498088719, 0.8351853498088719, 0.8353631433905236, 0.8352742465996977, 0.8353631433905236, 0.8349186594363943, 0.835096453018046, 0.8352742465996977, 0.8349186594363943, 0.8345630722730909, 0.8344741754822651, 0.8337630011556583, 0.8337630011556583, 0.832696239665748, 0.8328740332473997, 0.8326073428749222, 0.8323406525024447, 0.8322517557116188, 0.8320739621299671, 0.8318072717574896, 0.8318961685483154, 0.8324295492932705, 0.8326073428749222, 0.8322517557116188, 0.8327851364565739, 0.8311849942217086, 0.831540581385012, 0.8294959551960174, 0.8300293359409725, 0.8286958840785847, 0.8287847808694107, 0.8266512578895902, 0.8271846386345453, 0.8247844252822473, 0.8254955996088541, 0.8222953151391235, 0.8227397990932528, 0.8193617210418704, 0.8197173082051737]
In [105]:
# plot the results (Manhattan distance)
plt.plot(k_potentials, grid_mean_scores)
plt.xlabel('Potential K values for KNN')
plt.ylabel('Cross-Validated Accuracy')
Out[105]:
<matplotlib.text.Text at 0x120cb5a50>

In [106]:
# plot the results (using Euclidean distance)
plt.plot(k_potentials, grid_mean_scores)
plt.xlabel('Potential K values for KNN')
plt.ylabel('Cross-Validated Accuracy')
Out[106]:
<matplotlib.text.Text at 0x12143b7d0>

In [107]:
# examine the best model
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
0.962396657481
{'n_neighbors': 1}
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
metric_params=None, n_jobs=1, n_neighbors=1, p=2,
weights='uniform')
In [ ]:

In [108]:
# read in the data & create matrices
X = X_scaled[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", \
"time_spend_company", "Work_accident", "promotion_last_5years", "low", "medium",\
"technical", "support", "IT", "product_mng", "marketing", "RandD", "accounting",\
"hr"]]
y = X_scaled["left"]
In [176]:
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
Support Vector Machines
In [177]:
# Train, fit and predict with SVM

from sklearn import svm
Model=svm.SVC(kernel='linear')
Model.fit(X_train,y_train)
Y_pred=Model.predict(X_test)
In [178]:
# Cross validation score for SVM
scores = cross_val_score(Model, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.780694712068
In [179]:
# Evaluate SVM for accuracy

from sklearn import metrics
metrics.accuracy_score(y_test,Y_pred)
Out[179]:
0.78106666666666669
In [180]:
# Confusion matrix for SVM

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2696  166]
[ 655  233]]
In [181]:
# Classification Accuracy confirmation between manual calculation and formula for SVM

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.781066666667
0.781066666667
In [182]:
# Misclassification Accuracy confirmation between manual calculation and formula for SVM
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.218933333333
0.218933333333
In [183]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated SVM

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.262387387387
0.262387387387
In [184]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for SVM

print(TN / float(TN + FP))
0.941998602376
In [185]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for SVM

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.583959899749
0.583959899749
In [186]:
# Misclassification Accuracy confirmation between manual calculation and formula for SVM
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.218933333333
0.218933333333
In [187]:
# Roc_curve for Support Vector Machine Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [92]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [93]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Support Vector Machines: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

Random Forest
In [157]:
# Random Forests Classifier from Scikit Learn

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_jobs=2)
RF.fit(X_train,y_train)
Y_pred=RF.predict(X_test)
In [158]:
# Score for 10-fold Cross Validation for Random Forests Classifier

scores = cross_val_score(RF, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.986309602501
In [159]:
# Prediction Accuracy for Random Forests

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.9896
In [160]:
# Confusion matrix for Random Forests

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2857    5]
[  34  854]]
In [161]:
# Classification Accuracy confirmation between manual calculation and formula for Random Forest

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.9896
0.9896
In [162]:
# Misclassification Accuracy confirmation between manual calculation and formula for Random Forest
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0104
0.0104
In [163]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for Random Forest Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.961711711712
0.961711711712
In [164]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for Random Forest Classifier

print(TN / float(TN + FP))
0.998252969951
In [165]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for Random Forest Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.994179278231
0.994179278231
In [166]:
# Misclassification Accuracy confirmation between manual calculation and formula for Random Forest
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0104
0.0104
In [167]:
# Roc_curve for Random Forest Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [168]:
y_pred_prob_Random_Forest = RF.predict_proba(X_test)[:, 1]
In [172]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_Random_Forest, bins=5)
plt.xlim(0, 1)
plt.title('Random Forest Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[172]:
<matplotlib.text.Text at 0x12013a8d0>

In [170]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [171]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Random Forest: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

Bagging
In [109]:
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
In [110]:
# Bagging Classifier

seed = 7
cart = DecisionTreeClassifier()
num_trees = 100
Bagging = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
Bagging.fit(X_train,y_train)
Y_pred=Bagging.predict(X_test)
In [111]:
# Cross Validation for Bagging

scores = cross_val_score(Bagging, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.987910076856
In [112]:
# Prediction Accuracy for Bagging

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.989333333333
In [113]:
# Confusion matrix for Bagging

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2854    8]
[  32  856]]
In [114]:
# Classification Accuracy confirmation between manual calculation and formula for Bagging

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.989333333333
0.989333333333
In [115]:
# Misclassification Accuracy confirmation between manual calculation and formula for Bagging
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0106666666667
0.0106666666667
In [116]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for Bagging Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.963963963964
0.963963963964
In [117]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for Bagging Classifier

print(TN / float(TN + FP))
0.997204751922
In [118]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for Bagging Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.990740740741
0.990740740741
In [119]:
# Misclassification Accuracy confirmation between manual calculation and formula for Bagging
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0106666666667
0.0106666666667
In [120]:
# Roc_curve for Bagging


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [121]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
In [122]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Bagging: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [123]:
y_pred_prob_Bagging = Bagging.predict_proba(X_test)[:, 1]
In [124]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_Bagging, bins=5)
plt.xlim(0, 1)
plt.title('Bagging Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[124]:
<matplotlib.text.Text at 0x11b217610>

In [ ]:

AdaBoost
In [125]:
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
In [126]:
# AdaBoost Classifier

seed = 7
cart = AdaBoostClassifier()
num_trees = 100
AdaBoostClassifier = AdaBoostClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
AdaBoostClassifier.fit(X_train,y_train)
Y_pred=AdaBoostClassifier.predict(X_test)
In [127]:
# Cross Validation for AdaBoost

scores = cross_val_score(AdaBoostClassifier, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.958576016468
In [128]:
# Prediction Accuracy for AdaBoost

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.958133333333
In [129]:
# Confusion matrix for AdaBoost

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2793   69]
[  88  800]]
In [130]:
# Classification Accuracy confirmation between manual calculation and formula for Bagging

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.958133333333
0.958133333333
In [131]:
# Misclassification Accuracy confirmation between manual calculation and formula for AdaBoost
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0418666666667
0.0418666666667
In [132]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for AdaBoost Boosting Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.900900900901
0.900900900901
In [133]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for AdaBoost Classifier

print(TN / float(TN + FP))
0.975890985325
In [134]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for AdaBoost Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.920598388953
0.920598388953
In [135]:
# Misclassification Accuracy confirmation between manual calculation and formula for AdaBoost
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0418666666667
0.0418666666667
In [136]:
# Roc_curve for AdaBoost


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [137]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
In [138]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='AdaBoost: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [139]:
y_pred_prob_AdaBoost = AdaBoostClassifier.predict_proba(X_test)[:, 1]
In [140]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_AdaBoost, bins=5)
plt.xlim(0, 1)
plt.title('AdaBoost Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[140]:
<matplotlib.text.Text at 0x11b367a10>

In [ ]:

In [ ]:

In [ ]:

Gradient Boosting
In [141]:
from sklearn.ensemble import GradientBoostingClassifier
In [142]:
# Train and predict Gradient Boosting

GradientBoosting = GradientBoostingClassifier()
GradientBoosting.fit(X_train,y_train)
Y_pred=GradientBoosting.predict(X_test)
In [143]:
# Cross Validation for Gradient Boosting

scores = cross_val_score(GradientBoosting, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.976443246546
In [144]:
# Prediction Accuracy for Gradient Boosting

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.974133333333
In [145]:
# Confusion matrix for Gradient Boosting

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2832   30]
[  67  821]]
In [146]:
# Classification Accuracy confirmation between manual calculation and formula for Gradient Boosting

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.974133333333
0.974133333333
In [147]:
# Misclassification Accuracy confirmation between manual calculation and formula for Gradient Boosting
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0258666666667
0.0258666666667
In [148]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for Gradient Boosting Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.92454954955
0.92454954955
In [149]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for Gradient Classifier

print(TN / float(TN + FP))
0.989517819706
In [150]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for Gradient Boosting Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.964747356052
0.964747356052
In [151]:
# Misclassification Accuracy confirmation between manual calculation and formula for Gradient Boosting
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0258666666667
0.0258666666667
In [152]:
# Roc_curve for Gradient Boosting


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Gradient Boosting Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [153]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
In [154]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Gradient Boosting: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [155]:
y_pred_prob_GradientBoosting = GradientBoosting.predict_proba(X_test)[:, 1]
In [156]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_GradientBoosting, bins=5)
plt.xlim(0, 1)
plt.title('Gradient Boosting Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[156]:
<matplotlib.text.Text at 0x11c2566d0>

In [ ]:

In [ ]:

Part IV
Conclusion
One interesting chart showed us which departments have the most people leaving (and staying). They are Sales followed by Technical and Support. That means that the biggest effort at the company should be placed on keeping the strong employees in those departments (but by no means neglect the other departments).
There are several other interesting patterns in the data. Some are very intuitive whereas others very surprising and unexpected. There are three clusters of people leaving the company (the most important cluster is not as dense, in the sense that there are actually slightly more people there that stay than leave, but nevertheless its the most interesting and surprising cluster).
The three groups are employees with high performance and low satisfaction, employees with low satisfcation and performance (rather obvious) and employees with high satisfaction and performance (a surprising result). The first two results are very strong but rather trivial, there is limited use in providing detailed analyses of them. However, the somewhat less dense but very significant group of highly satisfied employees that do remarkably well in their evaluations is of extreme importance. Why do they leave? Well, there are several important factors to consider.
The people with high salaries and at even average satisfaction levels rarely leave the firm. Hence, one attribute that people who tend to leave will generally possess is salaries that are not high and fall in the low or medium range. It is interesting to note though that people with near perfect to perfect satisfaction levels do not leave even if they are in the low or medium income range brackets. Hence, a possible course of action to potentially alleviate this issue is increase salaries for good employees. Also, a further investigation into what might be the differentiating factor between people who are very satisfied (0.7-0.9 rate) vs super satisfied (0.9-1.0) can be done to see if it is possible to further increase the satisfaction level of those that are very satisfied but not perfectly so (maybe its possible to do this while at the same time saving money by not instituting as much pay increases if there is a way to get them satisfied in another manner, perhaps by working less hours for instance) would be a good idea.
Secondly, promotions are a huge factor. Looking at the plot of promotions during the past five years, it is very clear that people who are promoted stay and people who are not promoted leave. Of course, promotions and salary increases have some correlation because people tend to receive salary increases during promotions. Hence, it's important to promote employees if they do good work! That will help decrease the turnover rate.
The third important point is that employees who work a lot of hours that stay at the company for at least a medium amount of time tend to leave. Only the relatively small number of employees who work at the company for many years stay at the company regardless of working hours but very few make it there. Hence, the hours worked plays an important decision making role when employees think about whether to stay at the firm or seek other employment. Due to this finding it is important to see if any kind of rebalancing can be done so that part of the work can perhaps be shifted between employees who do not work as many hours and those that do in order to make the distribution of hours worked by employees more uniform which can perhaps increase employees willingness to stay at the company longer. It is quiet possible that some employees just feel burnt out even if they are generally happy with their job. Due to an inefficient work life balance, they might seek to find employment elsewhere.
Multiple machine learning algorithms were ran on the data and quiet a few did relatively well in terms of making predictions on out of sample data as measured by the accuracy score, the confusion matrix and deriatives thereof as well as the ROC Curve. In particular, the Random Forest and Bagging algorithms did the best with accuracy and cross-validation scores marginally different from each other just under 99%. Both correctly predicted over 3,700 out of 3,750 employees accurately (compared to a baseline accuracy of only 76% (2,850) as to whether they would leave/stay). The various scores computed from the confusion matrix were all almost perfect. Particularly importantly, the ROC curve has an area covering of .98 for both those algorithms.
Hence, using either of these two algorithms (Random Forests or Bagging) have given us great predictions as to which employees will leave vs which ones would stay.
Assumptions
We have assumed the dataset is accurate as we have not audited its validity.
Another assumption we made is that we trained on 75% of the data, conducted cross-validation, and then tested on the 25% remaining out of sample. However, when trying to sample to train a larger sample of the data (such as 90/10 test/train split), we received very similar results in our algorithms. Hence, our model is robust as should perform well when used on future data.
An assumption made by the Random Forest Model which did exceptionally well is that the sampling that it conducted during boostrap aggregation is actually representative. This is a safe assumption to make and the model has done really well on out of sample data.
Introduction:
This project analyzes a dataset in an attempt to determine what causes the good employees to leave their current jobs. The first step is to clean the data (Part I) and visualize it (Part II) using various types of graphs and charts to help establish clear relationships between various features. Part III analyzes the dataset using various Machine Learning algorithms and picks the best algorithm to model this dataset based on various metrics. Lastly, a conclusion on the entire analysis is provided.
Part I
Data Cleaning
This initial stage of the project is to clean the data in such a way that all the important attributes of the dataset be retained and unhindered for analysis purposes. Hence, the initial cleaning phase of the project imports the dataset and the libraries needed for the cleaning stages of the project. A check for Null Values/Missing Values is conducted (none are present). The multiple categorical variables in the dataset are created into dummy variables which is necessary for later stages of the analysis. These new variables are afterwards added to the original dataset and the initial variables from which dummification was done are removed to avoid redundancy. Lastly, the dataset is normalized due to the need to apply machine learning algorithms to it in later stages of the project.
In [1]:
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

%matplotlib inline
In [2]:
HR = pd.read_csv('HR_comma_sep.csv')
HR.head(10)
Out[2]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary
0	0.38	0.53	2	157	3	0	1	0	sales	low
1	0.80	0.86	5	262	6	0	1	0	sales	medium
2	0.11	0.88	7	272	4	0	1	0	sales	medium
3	0.72	0.87	5	223	5	0	1	0	sales	low
4	0.37	0.52	2	159	3	0	1	0	sales	low
5	0.41	0.50	2	153	3	0	1	0	sales	low
6	0.10	0.77	6	247	4	0	1	0	sales	low
7	0.92	0.85	5	259	5	0	1	0	sales	low
8	0.89	1.00	5	224	5	0	1	0	sales	low
9	0.42	0.53	2	142	3	0	1	0	sales	low
In [3]:
HR[HR['satisfaction_level'] == np.nan]
HR[HR['last_evaluation'] == np.nan]
HR[HR['number_project'] == np.nan]
HR[HR['average_montly_hours'] == np.nan]
HR[HR['time_spend_company'] == np.nan]
HR[HR['Work_accident'] == np.nan]
HR[HR['left'] == np.nan]
HR[HR['sales'] == np.nan]
HR[HR['salary'] == np.nan]
Out[3]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary
In [4]:
# Create a heatmap, to check for null values.

plt.figure(figsize=(10,7))
sns.heatmap(HR.isnull(),yticklabels=False, cbar=False, cmap='viridis')
Out[4]:
<matplotlib.axes._subplots.AxesSubplot at 0x114c6ee50>

In [5]:
HR['sales'].value_counts()
Out[5]:
sales          4140
technical      2720
support        2229
IT             1227
product_mng     902
marketing       858
RandD           787
accounting      767
hr              739
management      630
Name: sales, dtype: int64
In [6]:
HR['salary'].value_counts()
Out[6]:
low       7316
medium    6446
high      1237
Name: salary, dtype: int64
In [7]:
HR['left'].value_counts()
Out[7]:
0    11428
1     3571
Name: left, dtype: int64
In [8]:
# Create dummy variables for all the departments

sales = pd.get_dummies(HR.sales)

sales.columns = ["IT", "RandD", "accounting", "hr", "management", "marketing", "product_mng", "sales_department", "support", "technical"]

sales.head()
Out[8]:
IT	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical
0	0	0	0	0	0	0	0	1	0	0
1	0	0	0	0	0	0	0	1	0	0
2	0	0	0	0	0	0	0	1	0	0
3	0	0	0	0	0	0	0	1	0	0
4	0	0	0	0	0	0	0	1	0	0
In [9]:
# Create dummy variables for the salary

salary = pd.get_dummies(HR.salary)

salary.columns = ['low', 'medium', 'high']

salary.head()
Out[9]:
low	medium	high
0	0	1	0
1	0	0	1
2	0	0	1
3	0	1	0
4	0	1	0
In [10]:
# Create dummy variable for stayed/left

left = pd.get_dummies(HR.left)

left.columns = ['left', 'stayed']

left.head(10)
Out[10]:
left	stayed
0	0	1
1	0	1
2	0	1
3	0	1
4	0	1
5	0	1
6	0	1
7	0	1
8	0	1
9	0	1
In [11]:
# Concatenate the original dataset with the newly created salary and sales dummified variables

HR = pd.concat([HR, salary, sales], axis = 1)

HR.head(10)
Out[11]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary	...	IT	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical
0	0.38	0.53	2	157	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
1	0.80	0.86	5	262	6	0	1	0	sales	medium	...	0	0	0	0	0	0	0	1	0	0
2	0.11	0.88	7	272	4	0	1	0	sales	medium	...	0	0	0	0	0	0	0	1	0	0
3	0.72	0.87	5	223	5	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
4	0.37	0.52	2	159	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
5	0.41	0.50	2	153	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
6	0.10	0.77	6	247	4	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
7	0.92	0.85	5	259	5	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
8	0.89	1.00	5	224	5	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
9	0.42	0.53	2	142	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
10 rows × 23 columns
In [12]:
HR.dtypes
Out[12]:
satisfaction_level       float64
last_evaluation          float64
number_project             int64
average_montly_hours       int64
time_spend_company         int64
Work_accident              int64
left                       int64
promotion_last_5years      int64
sales                     object
salary                    object
low                        uint8
medium                     uint8
high                       uint8
IT                         uint8
RandD                      uint8
accounting                 uint8
hr                         uint8
management                 uint8
marketing                  uint8
product_mng                uint8
sales_department           uint8
support                    uint8
technical                  uint8
dtype: object
In [13]:
# Create a new column 

salary_map = {'low': 1, 'medium': 2, 'high': 3}

HR['salary_variable'] = HR['salary'].apply(lambda x: salary_map[x])

HR.head()
Out[13]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary	...	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical	salary_variable
0	0.38	0.53	2	157	3	0	1	0	sales	low	...	0	0	0	0	0	0	1	0	0	1
1	0.80	0.86	5	262	6	0	1	0	sales	medium	...	0	0	0	0	0	0	1	0	0	2
2	0.11	0.88	7	272	4	0	1	0	sales	medium	...	0	0	0	0	0	0	1	0	0	2
3	0.72	0.87	5	223	5	0	1	0	sales	low	...	0	0	0	0	0	0	1	0	0	1
4	0.37	0.52	2	159	3	0	1	0	sales	low	...	0	0	0	0	0	0	1	0	0	1
5 rows × 24 columns
In [14]:
del HR['salary']

del HR['sales']
In [15]:
HR.head()
Out[15]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	low	medium	...	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical	salary_variable
0	0.38	0.53	2	157	3	0	1	0	0	1	...	0	0	0	0	0	0	1	0	0	1
1	0.80	0.86	5	262	6	0	1	0	0	0	...	0	0	0	0	0	0	1	0	0	2
2	0.11	0.88	7	272	4	0	1	0	0	0	...	0	0	0	0	0	0	1	0	0	2
3	0.72	0.87	5	223	5	0	1	0	0	1	...	0	0	0	0	0	0	1	0	0	1
4	0.37	0.52	2	159	3	0	1	0	0	1	...	0	0	0	0	0	0	1	0	0	1
5 rows × 22 columns
In [16]:
HR.dtypes
Out[16]:
satisfaction_level       float64
last_evaluation          float64
number_project             int64
average_montly_hours       int64
time_spend_company         int64
Work_accident              int64
left                       int64
promotion_last_5years      int64
low                        uint8
medium                     uint8
high                       uint8
IT                         uint8
RandD                      uint8
accounting                 uint8
hr                         uint8
management                 uint8
marketing                  uint8
product_mng                uint8
sales_department           uint8
support                    uint8
technical                  uint8
salary_variable            int64
dtype: object
In [17]:
from sklearn.preprocessing import MinMaxScaler
X_scaled = MinMaxScaler().fit_transform(HR.astype(float))
X_scaled = pd.DataFrame(X_scaled, columns = HR.columns)

# Min Max Scaler

X_scaled.head(10)
Out[17]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	low	medium	...	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical	salary_variable
0	0.318681	0.265625	0.0	0.285047	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
1	0.780220	0.781250	0.6	0.775701	0.500	0.0	1.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.5
2	0.021978	0.812500	1.0	0.822430	0.250	0.0	1.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.5
3	0.692308	0.796875	0.6	0.593458	0.375	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
4	0.307692	0.250000	0.0	0.294393	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
5	0.351648	0.218750	0.0	0.266355	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
6	0.010989	0.640625	0.8	0.705607	0.250	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
7	0.912088	0.765625	0.6	0.761682	0.375	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
8	0.879121	1.000000	0.6	0.598131	0.375	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
9	0.362637	0.265625	0.0	0.214953	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
10 rows × 22 columns
Part II
Exploratory Data Analysis (EDA)
In this section, we try to visualize the data on employees using various plots, graphs, charts in order to get an idea of how different features in the dataset relate to each other.
The below represents a heatmap of correlations of features. Some interesting observations from this heatmap: Quiet intuitively, you can see that the relationship between satisfaction level and people that left the company is negative. It's also interesting to note that there is a positive correlation between the number of hours employees work and their evaluation. These and other relationships will be explored more deeply in the rest of the EDA.
In [18]:
correlation = X_scaled.corr()
plt.figure(figsize=(20,20))
sns.heatmap(correlation, vmax=10, square=True,annot=True,cmap=None, linewidths=1)

plt.title('Correlation between features')
Out[18]:
<matplotlib.text.Text at 0x118f3a050>

In [19]:
# Exported csv file to work with in Tableau
X_scaled.to_csv("Updated_Capstone.csv")
In [20]:
X_scaled['left'].value_counts()
Out[20]:
0.0    11428
1.0     3571
Name: left, dtype: int64
In [21]:
stayed = np.where(X_scaled['left'] == 0)[0]
left = np.where(X_scaled['left'] == 1)[0]
For a given level of satisfaction, there is definitely an intuitive pattern that the higher someone's salary is, the less they are likely to leave. This relationship is a lot more apparent in the distinction between the high income earners vs low & median income earners as a group. The difference in the people that left between low and median income salaries is not significant. Hence, most of the people that leave are not receiving a high salary, which means that you if you would like to retain these employees, you should increase their pay!
In [22]:
# plot of Satisfaction vs Last Evaluation
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.satisfaction_level[stayed], X_scaled.salary_variable[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.satisfaction_level[left], X_scaled.salary_variable[left], alpha=0.1, color='r')
ax.set_title("satisfaction_level vs salary")
ax.set_xlabel("satisfaction_level")
ax.set_ylabel("salary")
plt.show()

In the Satisfaction vs Last Evaluation plot below, we can note some interesting observations. Firstly, there is a sizeable cluster of people with strong evaluations and low satisfaction levels leaving. Secondly, there is a significant group of people leaving that are slighly below average in performance and evaluation. Lastly, and by far the most interesting observation of the three, there is a decent size of a not very dense cluster of people leaving that have done extremely well on their evaluations (0.7 to 1 on a 0-1 scale) and and have very high satisfaction rates, albeit not perfect (0.7-0.9 on a 0-1 scale). That particular group of employees also have many people that stay, but a significant proportion of that group is leaving. So the most surprising thing to observe is that people can be satisfied with the job and still leave! There are several possibilites for why this might be possible. People might enjoy their job but think they can do even better elsewhere, whether its income or career growth (or a combination of the two, considering that those things tend to go hand in hand). Another possibility is that people might not be honest in surveys because of fear of being reprimanded for negative surveys.
In [23]:
# plot of Satisfaction vs Last Evaluation
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.satisfaction_level[stayed], X_scaled.last_evaluation[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.satisfaction_level[left], X_scaled.last_evaluation[left], alpha=0.1, color='r')
ax.set_title("satisfaction_level vs last_evaluation")
ax.set_xlabel("satisfaction_level")
ax.set_ylabel("last_evaluation")
plt.show()

The Satisfaction vs Promotion during the last five years plot below shows a very strong relationship in several interesting ways! People who have been promoted don't leave very often! However, such promotions are not common enough! Therefore, if you don't want good employees to leave, promoting them will really help!
In [24]:
# plot of Satisfaction vs Promotion during last five years
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.satisfaction_level[stayed], X_scaled.promotion_last_5years[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.satisfaction_level[left], X_scaled.promotion_last_5years[left], alpha=0.1, color='r')
ax.set_title("satisfaction_level vs promotion_last_5years")
ax.set_xlabel("satisfaction_level")
ax.set_ylabel("promotion_last_5years")
plt.show()

The below scatter plot compares the average working hours against the time spent at the company and how it impacts the employee turnover rate. There are multiple interesting relationships to note. For employees that just started out, the majority tend to stick around for some time even if they work substantial hours. However, for employees who have been at the company for a while, there is a significant tendency to leave the company if working substantial hours. Consequently, we can observe that very few employees stick around at the company for a long time period because the ones who were working long hours almost all left! However, those that do stick around for a long time (relatively few) tend to all be devoted and not leave
There are several conjectures for why this is possible. One is that employees have worked for a significant amount of hours and put in a lot of time and effort expecting a quick promotion and/or pay increase (refer to graph above for relationship between promotions and staying/leaving) that they have not received and leave for elsewhere. Another reason might be that people have tried working for a little bit of time, realized that they are being overworked and have started looking for other opportunities to seek employment.
In [25]:
# plot of Average_monthly_hours vs time_spend_company
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.average_montly_hours[stayed], X_scaled.time_spend_company[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.average_montly_hours[left], X_scaled.time_spend_company[left], alpha=0.1, color='r')
ax.set_title("average_monthly_hours vs time_spend_company")
ax.set_xlabel("average_monthly_hours")
ax.set_ylabel("time_spend_company")
plt.show()

In [26]:
# Imported from Tableau
from IPython.display import Image
Image(filename = 'Average Monthly Hours vs Satisfaction Level.png', width = 1000)
Out[26]:

There does not appear to be a strong relationship between accidents and people leaving the company, hence, people stay/leave regardless of the accidents they might have experienced.
In [27]:
# plot of work_accident vs time_spend_company
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.Work_accident[stayed], X_scaled.time_spend_company[stayed], alpha=0.5, color='g')
ax.scatter(X_scaled.Work_accident[left], X_scaled.time_spend_company[left], alpha=0.5, color='r')
ax.set_title("Work_accident vs time_spend_company")
ax.set_xlabel("Work_accident")
ax.set_ylabel("time_spend_company")
plt.show()

The below stripplot compares number of projects done by an employee versus the number of promotions they have received over the past 5 years. It is interesting to note that the number of projects completed has very insignificant impact on whether a person leaves. On the contrary, the promotions received during the past 5 years is of critical importance. The majority of people who have received a promotion stay. Almost all those who were not promoted leave.
The data seems insufficient to be able to determine why the number of projects is not related to promotions. There are at least several possibilities. Perhaps the number of projects refers to the number assigned (vs completed) or that the projects are not all of equal difficulty and not equally time consuming leading to distorted hypothesis that the number of projects might necessarily correspond to an employee's productivity.
In [28]:
import seaborn as sns

sns.stripplot("number_project", "promotion_last_5years", data=X_scaled, hue="left", jitter = True)
Out[28]:
<matplotlib.axes._subplots.AxesSubplot at 0x117949d90>

The Satisfaction level distribution below shows that the majority of people are at least moderately satisfied with their job. Therefore, for the people that are satisfied and do well, efforts should be placed on making them even more happy by promoting them. For the people that do poorly or mediocre, either studies should be placed on determining why they are not doing well and trying to improve their performance or attempting to replace them with better employees (this would depend on a cost/benefit analysis) done by the company regarding this issue.
In [29]:
# Histogram distribution of Satisfaction Level

f,ax1 = plt.subplots(1,1)
sns.distplot(X_scaled['satisfaction_level'],bins=10,norm_hist=False)
plt.show()

The distribution (which is a decent approximation of a normal distribution with almost no tails and some kurtosis present) of the last evaluation graph shows us there is a sizeable portion of people that do well in their evaluations, hence the company should try to do as much as possible to keep those people!
In [30]:
# Histogram distribution of Last Evaluation


f,ax1 = plt.subplots(1,1)
sns.distplot(X_scaled['last_evaluation'],bins=10,norm_hist=False)
plt.show()

The below plot shows the ratios of people who stay in the company by department and the ratio of people who leave the company by department. It is clear from the chart that the departments with the most number of employees also have the highest ratios of both people who have left and those that stayed at the company. This tells us there should be particular effort place on retaining good employees that work in those departments, in particular the sales, technical and support departments, respectively. The largest efforts should be placed on retaining the good employees in these groups because they have the most people leaving.
In [34]:
# This plot is to compare the people that left across different departments
# The non-department variables are deleted to only keep the departments

del X_scaled['satisfaction_level']
del X_scaled['last_evaluation']
del X_scaled['number_project']
del X_scaled['average_montly_hours']
del X_scaled['time_spend_company']
del X_scaled['Work_accident']
del X_scaled['promotion_last_5years']
del X_scaled['low']
del X_scaled['medium']
del X_scaled['high']
del X_scaled['salary_variable']


left=X_scaled.groupby('left').mean()
left=left.transpose()
plt.figure(figsize=(20,10))
left.plot(kind='bar')
Out[34]:
<matplotlib.axes._subplots.AxesSubplot at 0x11a7bcd10>
<matplotlib.figure.Figure at 0x11a360850>

Part III
Predictive Modelling
In this section below, multiple models will be run that are applicable to classification. Afterwards, various scoring metrics will be calculated for the models and a determination will be made as to which model is based for this dataset.
In [174]:
# read in the data & create matrices
X = X_scaled[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", \
"time_spend_company", "Work_accident", "promotion_last_5years", "salary_variable",\
"technical", "support", "IT", "product_mng", "marketing", "RandD", "accounting",\
"hr", "management", "sales_department"]]
y = X_scaled["left"]
In [175]:
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
Logistic Regression Model
In [33]:
# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
model = LogisticRegression()

# fit the model with data
mdl = model.fit(X_train, y_train)

# predict the response for new observations

logit = model.predict(X_test)

# sklearn output to check that ran on right data split.

len(logit)
Out[33]:
3750
In [34]:
# These represent the y-intercept and coefficients for all the variables in the logistic regression.

print(model.intercept_)
print(model.coef_)
[ 0.57305398]
[[-3.71487625  0.45484967 -1.50619896  0.8704181   1.97590531 -1.45425383
-1.26024249 -1.32512177  0.22588571  0.23497924 -0.01282827  0.07509077
0.13925062 -0.47542127  0.26606552  0.3920348  -0.44451555  0.17251241]]
In [35]:
# Prediction Accuracy for Logistic Regression

from sklearn import metrics
print(metrics.accuracy_score(y_test, logit))
0.796
In [36]:
from sklearn.cross_validation import cross_val_score
In [37]:
# 10-fold cross-validation for Logistic Regression to find the CV score
mdl = model.fit(X_train, y_train)
scores = cross_val_score(mdl, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.794205514996
In [38]:
from sklearn import metrics
confusion = metrics.confusion_matrix(y_test, logit)
print ((confusion))
[[2651  211]
[ 554  334]]
In [39]:
y_pred_prob = model.predict_proba(X_test)[:, 1]
In [40]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob, bins=5)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[40]:
<matplotlib.text.Text at 0x1181a6a90>

In [41]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [42]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, logit)


plot_confusion_matrix(cm, title='LogisticRegression: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [43]:
# Roc_curve for Logistic Regression Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, logit)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Logistic Regression Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

K Nearest Nearbours Model
K=1
In [44]:
from sklearn.neighbors import KNeighborsClassifier
In [45]:
# K-Nearest Neighbours with K=1
knn = KNeighborsClassifier(n_neighbors=1, metric = 'euclidean')
In [46]:
y = knn.fit(X_train, y_train)

KNN = knn.predict(X_test)

# sklearn output to check that ran on right data split.

len(KNN)
Out[46]:
3750
In [47]:
# Cross fold validation for K=1

md2 = knn.fit(X_train, y_train)
scores = cross_val_score(md2, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.962043001151
In [48]:
# Prediction Accuracy.

from sklearn import metrics
print(metrics.accuracy_score(y_test, KNN))
0.967466666667
In [49]:
confusion = metrics.confusion_matrix(y_test, KNN)
print confusion
[[2775   87]
[  35  853]]
In [50]:
# Classification Accuracy confirmation between manual calculation and formula for K=1

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, KNN))
0.967466666667
0.967466666667
In [51]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=1
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN))
0.0325333333333
0.0325333333333
In [52]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for K=1

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, KNN))
0.960585585586
0.960585585586
In [53]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for K=1


print(TN / float(TN + FP))
0.969601677149
In [54]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for K=1

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, KNN))
0.907446808511
0.907446808511
In [55]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=1
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN))
0.0325333333333
0.0325333333333
In [56]:
y_pred_prob_knn = knn.predict_proba(X_test)[:, 1]
In [57]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_knn, bins=5)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[57]:
<matplotlib.text.Text at 0x1181d6b90>

In [58]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [59]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, KNN)


plot_confusion_matrix(cm, title='KNN for K=1: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [60]:
# Roc_curve for KNN=1 Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, KNN)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('KNN=1 Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

K=5
In [61]:
# K-Nearest Neighbours with K=5

knn = KNeighborsClassifier(n_neighbors=5, metric = 'euclidean')
y = knn.fit(X_train, y_train)
KNN5 = knn.predict(X_test)

# sklearn output to check that ran on right data split.

len(KNN5)
Out[61]:
3750
In [62]:
# Cross Validation for K=5 Cross-Fold Validation

md2 = knn.fit(X_train, y_train)
scores = cross_val_score(md2, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.938309179837
In [63]:
# Prediction Accuracy on KNN=5 for accuracy score

from sklearn import metrics
print(metrics.accuracy_score(y_test, KNN5))
0.941333333333
In [64]:
confusion = metrics.confusion_matrix(y_test, KNN5)
print confusion
[[2731  131]
[  89  799]]
In [65]:
# Classification Accuracy confirmation between manual calculation and formula for K=5

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, KNN5))
0.941333333333
0.941333333333
In [66]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=5
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN5))
0.0586666666667
0.0586666666667
In [67]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for K=5

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, KNN5))
0.899774774775
0.899774774775
In [68]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for K=5

print(TN / float(TN + FP))
0.954227812718
In [69]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for K=5

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, KNN5))
0.859139784946
0.859139784946
In [70]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=5
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN5))
0.0586666666667
0.0586666666667
In [71]:
# Roc_curve for KNN=5 Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, KNN5)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [72]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [73]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, KNN5)


plot_confusion_matrix(cm, title='KNN for N=5: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [74]:
y_pred_prob_knn = knn.predict_proba(X_test)[:, 1]
In [75]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_knn, bins=5)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[75]:
<matplotlib.text.Text at 0x11acdea10>

In [ ]:

Grid Search for optimal N value in KNN
In [101]:
from sklearn.grid_search import GridSearchCV
/Users/Misha/anaconda/lib/python2.7/site-packages/sklearn/grid_search.py:43: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.
DeprecationWarning)
In [102]:
k_potentials = list(range(1, 200))
print(k_potentials)
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]
In [103]:
# Set Nearest Neighbours equal to k
knn = KNeighborsClassifier(n_neighbors=k_potentials, metric = 'euclidean')
# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_potentials)
# instantiate the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
# fit the grid with data
grid.fit(X_train, y_train)
# view the results
grid.grid_scores_
Out[103]:
[mean: 0.96240, std: 0.00591, params: {'n_neighbors': 1},
mean: 0.95884, std: 0.00543, params: {'n_neighbors': 2},
mean: 0.94568, std: 0.00463, params: {'n_neighbors': 3},
mean: 0.94622, std: 0.00555, params: {'n_neighbors': 4},
mean: 0.93875, std: 0.00620, params: {'n_neighbors': 5},
mean: 0.94284, std: 0.00673, params: {'n_neighbors': 6},
mean: 0.93795, std: 0.00746, params: {'n_neighbors': 7},
mean: 0.93839, std: 0.00819, params: {'n_neighbors': 8},
mean: 0.93351, std: 0.00932, params: {'n_neighbors': 9},
mean: 0.93493, std: 0.00848, params: {'n_neighbors': 10},
mean: 0.93048, std: 0.00825, params: {'n_neighbors': 11},
mean: 0.93253, std: 0.00810, params: {'n_neighbors': 12},
mean: 0.92933, std: 0.00898, params: {'n_neighbors': 13},
mean: 0.92968, std: 0.00824, params: {'n_neighbors': 14},
mean: 0.92719, std: 0.00747, params: {'n_neighbors': 15},
mean: 0.92817, std: 0.00830, params: {'n_neighbors': 16},
mean: 0.92559, std: 0.00873, params: {'n_neighbors': 17},
mean: 0.92728, std: 0.00821, params: {'n_neighbors': 18},
mean: 0.92497, std: 0.00821, params: {'n_neighbors': 19},
mean: 0.92470, std: 0.00830, params: {'n_neighbors': 20},
mean: 0.92204, std: 0.00804, params: {'n_neighbors': 21},
mean: 0.92186, std: 0.00867, params: {'n_neighbors': 22},
mean: 0.91902, std: 0.00856, params: {'n_neighbors': 23},
mean: 0.91777, std: 0.00850, params: {'n_neighbors': 24},
mean: 0.91564, std: 0.00809, params: {'n_neighbors': 25},
mean: 0.91457, std: 0.00913, params: {'n_neighbors': 26},
mean: 0.91244, std: 0.00927, params: {'n_neighbors': 27},
mean: 0.91244, std: 0.00873, params: {'n_neighbors': 28},
mean: 0.91101, std: 0.00878, params: {'n_neighbors': 29},
mean: 0.91110, std: 0.00853, params: {'n_neighbors': 30},
mean: 0.90906, std: 0.00855, params: {'n_neighbors': 31},
mean: 0.90941, std: 0.00906, params: {'n_neighbors': 32},
mean: 0.90773, std: 0.00922, params: {'n_neighbors': 33},
mean: 0.90728, std: 0.00991, params: {'n_neighbors': 34},
mean: 0.90550, std: 0.00946, params: {'n_neighbors': 35},
mean: 0.90417, std: 0.00891, params: {'n_neighbors': 36},
mean: 0.90292, std: 0.00887, params: {'n_neighbors': 37},
mean: 0.90097, std: 0.00881, params: {'n_neighbors': 38},
mean: 0.90035, std: 0.00948, params: {'n_neighbors': 39},
mean: 0.89875, std: 0.01009, params: {'n_neighbors': 40},
mean: 0.89759, std: 0.01009, params: {'n_neighbors': 41},
mean: 0.89626, std: 0.00955, params: {'n_neighbors': 42},
mean: 0.89537, std: 0.01016, params: {'n_neighbors': 43},
mean: 0.89270, std: 0.01221, params: {'n_neighbors': 44},
mean: 0.89172, std: 0.01244, params: {'n_neighbors': 45},
mean: 0.88950, std: 0.01089, params: {'n_neighbors': 46},
mean: 0.88906, std: 0.01150, params: {'n_neighbors': 47},
mean: 0.88666, std: 0.01137, params: {'n_neighbors': 48},
mean: 0.88577, std: 0.01143, params: {'n_neighbors': 49},
mean: 0.88523, std: 0.01132, params: {'n_neighbors': 50},
mean: 0.88452, std: 0.01149, params: {'n_neighbors': 51},
mean: 0.88372, std: 0.01237, params: {'n_neighbors': 52},
mean: 0.88283, std: 0.01227, params: {'n_neighbors': 53},
mean: 0.88141, std: 0.01166, params: {'n_neighbors': 54},
mean: 0.88114, std: 0.01085, params: {'n_neighbors': 55},
mean: 0.88061, std: 0.01041, params: {'n_neighbors': 56},
mean: 0.87990, std: 0.01105, params: {'n_neighbors': 57},
mean: 0.87910, std: 0.00974, params: {'n_neighbors': 58},
mean: 0.87892, std: 0.00997, params: {'n_neighbors': 59},
mean: 0.87803, std: 0.01108, params: {'n_neighbors': 60},
mean: 0.87768, std: 0.01119, params: {'n_neighbors': 61},
mean: 0.87688, std: 0.01152, params: {'n_neighbors': 62},
mean: 0.87652, std: 0.01182, params: {'n_neighbors': 63},
mean: 0.87590, std: 0.01129, params: {'n_neighbors': 64},
mean: 0.87554, std: 0.01148, params: {'n_neighbors': 65},
mean: 0.87474, std: 0.01183, params: {'n_neighbors': 66},
mean: 0.87412, std: 0.01223, params: {'n_neighbors': 67},
mean: 0.87394, std: 0.01196, params: {'n_neighbors': 68},
mean: 0.87341, std: 0.01210, params: {'n_neighbors': 69},
mean: 0.87314, std: 0.01231, params: {'n_neighbors': 70},
mean: 0.87270, std: 0.01219, params: {'n_neighbors': 71},
mean: 0.87226, std: 0.01168, params: {'n_neighbors': 72},
mean: 0.87154, std: 0.01185, params: {'n_neighbors': 73},
mean: 0.87074, std: 0.01104, params: {'n_neighbors': 74},
mean: 0.87057, std: 0.01132, params: {'n_neighbors': 75},
mean: 0.87003, std: 0.01129, params: {'n_neighbors': 76},
mean: 0.86968, std: 0.01148, params: {'n_neighbors': 77},
mean: 0.86754, std: 0.01056, params: {'n_neighbors': 78},
mean: 0.86692, std: 0.01041, params: {'n_neighbors': 79},
mean: 0.86568, std: 0.01028, params: {'n_neighbors': 80},
mean: 0.86594, std: 0.01024, params: {'n_neighbors': 81},
mean: 0.86461, std: 0.00937, params: {'n_neighbors': 82},
mean: 0.86390, std: 0.00915, params: {'n_neighbors': 83},
mean: 0.86239, std: 0.00956, params: {'n_neighbors': 84},
mean: 0.86257, std: 0.01001, params: {'n_neighbors': 85},
mean: 0.86088, std: 0.01023, params: {'n_neighbors': 86},
mean: 0.86088, std: 0.01051, params: {'n_neighbors': 87},
mean: 0.85981, std: 0.01214, params: {'n_neighbors': 88},
mean: 0.85937, std: 0.01208, params: {'n_neighbors': 89},
mean: 0.85830, std: 0.01087, params: {'n_neighbors': 90},
mean: 0.85821, std: 0.01108, params: {'n_neighbors': 91},
mean: 0.85616, std: 0.00995, params: {'n_neighbors': 92},
mean: 0.85652, std: 0.00954, params: {'n_neighbors': 93},
mean: 0.85519, std: 0.00892, params: {'n_neighbors': 94},
mean: 0.85545, std: 0.00957, params: {'n_neighbors': 95},
mean: 0.85368, std: 0.00900, params: {'n_neighbors': 96},
mean: 0.85332, std: 0.00845, params: {'n_neighbors': 97},
mean: 0.85199, std: 0.00923, params: {'n_neighbors': 98},
mean: 0.85190, std: 0.00926, params: {'n_neighbors': 99},
mean: 0.85021, std: 0.00825, params: {'n_neighbors': 100},
mean: 0.85039, std: 0.00930, params: {'n_neighbors': 101},
mean: 0.84834, std: 0.01015, params: {'n_neighbors': 102},
mean: 0.84825, std: 0.01016, params: {'n_neighbors': 103},
mean: 0.84559, std: 0.00954, params: {'n_neighbors': 104},
mean: 0.84559, std: 0.00922, params: {'n_neighbors': 105},
mean: 0.84416, std: 0.00977, params: {'n_neighbors': 106},
mean: 0.84452, std: 0.00978, params: {'n_neighbors': 107},
mean: 0.84372, std: 0.00935, params: {'n_neighbors': 108},
mean: 0.84399, std: 0.00922, params: {'n_neighbors': 109},
mean: 0.84256, std: 0.00943, params: {'n_neighbors': 110},
mean: 0.84239, std: 0.00930, params: {'n_neighbors': 111},
mean: 0.84132, std: 0.00941, params: {'n_neighbors': 112},
mean: 0.84087, std: 0.00924, params: {'n_neighbors': 113},
mean: 0.84034, std: 0.00864, params: {'n_neighbors': 114},
mean: 0.84016, std: 0.00875, params: {'n_neighbors': 115},
mean: 0.83981, std: 0.00849, params: {'n_neighbors': 116},
mean: 0.83972, std: 0.00862, params: {'n_neighbors': 117},
mean: 0.83865, std: 0.00779, params: {'n_neighbors': 118},
mean: 0.83892, std: 0.00782, params: {'n_neighbors': 119},
mean: 0.83883, std: 0.00798, params: {'n_neighbors': 120},
mean: 0.83892, std: 0.00810, params: {'n_neighbors': 121},
mean: 0.83661, std: 0.00577, params: {'n_neighbors': 122},
mean: 0.83705, std: 0.00582, params: {'n_neighbors': 123},
mean: 0.83670, std: 0.00582, params: {'n_neighbors': 124},
mean: 0.83714, std: 0.00545, params: {'n_neighbors': 125},
mean: 0.83732, std: 0.00565, params: {'n_neighbors': 126},
mean: 0.83741, std: 0.00610, params: {'n_neighbors': 127},
mean: 0.83732, std: 0.00647, params: {'n_neighbors': 128},
mean: 0.83741, std: 0.00634, params: {'n_neighbors': 129},
mean: 0.83759, std: 0.00626, params: {'n_neighbors': 130},
mean: 0.83750, std: 0.00609, params: {'n_neighbors': 131},
mean: 0.83759, std: 0.00639, params: {'n_neighbors': 132},
mean: 0.83767, std: 0.00616, params: {'n_neighbors': 133},
mean: 0.83732, std: 0.00640, params: {'n_neighbors': 134},
mean: 0.83776, std: 0.00641, params: {'n_neighbors': 135},
mean: 0.83723, std: 0.00690, params: {'n_neighbors': 136},
mean: 0.83794, std: 0.00675, params: {'n_neighbors': 137},
mean: 0.83687, std: 0.00710, params: {'n_neighbors': 138},
mean: 0.83687, std: 0.00718, params: {'n_neighbors': 139},
mean: 0.83607, std: 0.00744, params: {'n_neighbors': 140},
mean: 0.83625, std: 0.00725, params: {'n_neighbors': 141},
mean: 0.83501, std: 0.00739, params: {'n_neighbors': 142},
mean: 0.83536, std: 0.00761, params: {'n_neighbors': 143},
mean: 0.83430, std: 0.00774, params: {'n_neighbors': 144},
mean: 0.83536, std: 0.00776, params: {'n_neighbors': 145},
mean: 0.83510, std: 0.00746, params: {'n_neighbors': 146},
mean: 0.83527, std: 0.00732, params: {'n_neighbors': 147},
mean: 0.83412, std: 0.00716, params: {'n_neighbors': 148},
mean: 0.83492, std: 0.00704, params: {'n_neighbors': 149},
mean: 0.83474, std: 0.00740, params: {'n_neighbors': 150},
mean: 0.83483, std: 0.00759, params: {'n_neighbors': 151},
mean: 0.83492, std: 0.00751, params: {'n_neighbors': 152},
mean: 0.83536, std: 0.00788, params: {'n_neighbors': 153},
mean: 0.83492, std: 0.00833, params: {'n_neighbors': 154},
mean: 0.83492, std: 0.00841, params: {'n_neighbors': 155},
mean: 0.83492, std: 0.00815, params: {'n_neighbors': 156},
mean: 0.83510, std: 0.00822, params: {'n_neighbors': 157},
mean: 0.83527, std: 0.00826, params: {'n_neighbors': 158},
mean: 0.83554, std: 0.00832, params: {'n_neighbors': 159},
mean: 0.83554, std: 0.00778, params: {'n_neighbors': 160},
mean: 0.83519, std: 0.00745, params: {'n_neighbors': 161},
mean: 0.83519, std: 0.00757, params: {'n_neighbors': 162},
mean: 0.83536, std: 0.00715, params: {'n_neighbors': 163},
mean: 0.83527, std: 0.00767, params: {'n_neighbors': 164},
mean: 0.83536, std: 0.00778, params: {'n_neighbors': 165},
mean: 0.83492, std: 0.00759, params: {'n_neighbors': 166},
mean: 0.83510, std: 0.00790, params: {'n_neighbors': 167},
mean: 0.83527, std: 0.00820, params: {'n_neighbors': 168},
mean: 0.83492, std: 0.00775, params: {'n_neighbors': 169},
mean: 0.83456, std: 0.00746, params: {'n_neighbors': 170},
mean: 0.83447, std: 0.00782, params: {'n_neighbors': 171},
mean: 0.83376, std: 0.00770, params: {'n_neighbors': 172},
mean: 0.83376, std: 0.00731, params: {'n_neighbors': 173},
mean: 0.83270, std: 0.00737, params: {'n_neighbors': 174},
mean: 0.83287, std: 0.00736, params: {'n_neighbors': 175},
mean: 0.83261, std: 0.00701, params: {'n_neighbors': 176},
mean: 0.83234, std: 0.00676, params: {'n_neighbors': 177},
mean: 0.83225, std: 0.00719, params: {'n_neighbors': 178},
mean: 0.83207, std: 0.00718, params: {'n_neighbors': 179},
mean: 0.83181, std: 0.00746, params: {'n_neighbors': 180},
mean: 0.83190, std: 0.00741, params: {'n_neighbors': 181},
mean: 0.83243, std: 0.00808, params: {'n_neighbors': 182},
mean: 0.83261, std: 0.00827, params: {'n_neighbors': 183},
mean: 0.83225, std: 0.00819, params: {'n_neighbors': 184},
mean: 0.83279, std: 0.00779, params: {'n_neighbors': 185},
mean: 0.83118, std: 0.00688, params: {'n_neighbors': 186},
mean: 0.83154, std: 0.00684, params: {'n_neighbors': 187},
mean: 0.82950, std: 0.00716, params: {'n_neighbors': 188},
mean: 0.83003, std: 0.00689, params: {'n_neighbors': 189},
mean: 0.82870, std: 0.00649, params: {'n_neighbors': 190},
mean: 0.82878, std: 0.00704, params: {'n_neighbors': 191},
mean: 0.82665, std: 0.00624, params: {'n_neighbors': 192},
mean: 0.82718, std: 0.00654, params: {'n_neighbors': 193},
mean: 0.82478, std: 0.00895, params: {'n_neighbors': 194},
mean: 0.82550, std: 0.00846, params: {'n_neighbors': 195},
mean: 0.82230, std: 0.00974, params: {'n_neighbors': 196},
mean: 0.82274, std: 0.01041, params: {'n_neighbors': 197},
mean: 0.81936, std: 0.00926, params: {'n_neighbors': 198},
mean: 0.81972, std: 0.00963, params: {'n_neighbors': 199}]
In [104]:
# create a list of the mean scores only
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print(grid_mean_scores)
[0.9623966574806649, 0.9588407858476309, 0.9456840608054049, 0.94621744155036, 0.9387501111209885, 0.9428393634989777, 0.9379500400035559, 0.9383945239576851, 0.9335052004622633, 0.9349275491154769, 0.9304827095741843, 0.9325273357631789, 0.9293270512934483, 0.9296826384567517, 0.9271935283136279, 0.9281713930127122, 0.9255933860787625, 0.9272824251044537, 0.9249711085429816, 0.9247044181705041, 0.9220375144457285, 0.9218597208640769, 0.9190150235576495, 0.9177704684860877, 0.9156369455062672, 0.914570184016357, 0.9124366610365365, 0.9124366610365365, 0.911014312383323, 0.9111032091741488, 0.9090585829851542, 0.9094141701484576, 0.9077251311227664, 0.9072806471686372, 0.9055027113521202, 0.9041692594897324, 0.9029247044181705, 0.9009689750200017, 0.9003466974842208, 0.8987465552493555, 0.8975908969686194, 0.8962574451062316, 0.8953684771979732, 0.8927015734731976, 0.8917237087741132, 0.8895012890034669, 0.8890568050493377, 0.8866565916970397, 0.8857676237887813, 0.8852342430438261, 0.8845230687172193, 0.8837229975997867, 0.8828340296915281, 0.8814116810383145, 0.881144990665837, 0.8806116099208818, 0.879900435594275, 0.8791003644768424, 0.8789225708951907, 0.8780336029869322, 0.8776780158236288, 0.8768779447061961, 0.8765223575428926, 0.8759000800071117, 0.8755444928438083, 0.8747444217263757, 0.8741221441905948, 0.873944350608943, 0.8734109698639879, 0.8731442794915104, 0.8726997955373811, 0.8722553115832519, 0.871544137256645, 0.8707440661392124, 0.8705662725575607, 0.8700328918126056, 0.8696773046493022, 0.8675437816694818, 0.8669215041337007, 0.8656769490621389, 0.8659436394346164, 0.8646101875722286, 0.8638990132456218, 0.8623877678015823, 0.862565561383234, 0.8608765223575429, 0.8608765223575429, 0.8598097608676327, 0.8593652769135034, 0.8582985154235933, 0.8582096186327673, 0.8561649924437728, 0.8565205796070762, 0.8551871277446884, 0.8554538181171659, 0.853675882300649, 0.8533202951373455, 0.8519868432749578, 0.851897946484132, 0.8502089074584408, 0.8503867010400925, 0.8483420748510979, 0.848253178060272, 0.8455862743354965, 0.8455862743354965, 0.8441639256822828, 0.8445195128455862, 0.8437194417281536, 0.8439861321006311, 0.8425637834474176, 0.8423859898657658, 0.8413192283758556, 0.8408747444217264, 0.8403413636767713, 0.8401635700951195, 0.8398079829318161, 0.8397190861409903, 0.8386523246510801, 0.8389190150235577, 0.8388301182327318, 0.8389190150235577, 0.8366076984620855, 0.8370521824162148, 0.8366965952529114, 0.8371410792070406, 0.8373188727886923, 0.8374077695795181, 0.8373188727886923, 0.8374077695795181, 0.8375855631611698, 0.837496666370344, 0.8375855631611698, 0.8376744599519957, 0.8373188727886923, 0.8377633567428215, 0.8372299759978665, 0.8379411503244732, 0.8368743888345631, 0.8368743888345631, 0.8360743177171304, 0.8362521112987821, 0.8350075562272202, 0.8353631433905236, 0.8342963819006134, 0.8353631433905236, 0.835096453018046, 0.8352742465996977, 0.8341185883189617, 0.8349186594363943, 0.8347408658547426, 0.8348297626455685, 0.8349186594363943, 0.8353631433905236, 0.8349186594363943, 0.8349186594363943, 0.8349186594363943, 0.835096453018046, 0.8352742465996977, 0.8355409369721754, 0.8355409369721754, 0.8351853498088719, 0.8351853498088719, 0.8353631433905236, 0.8352742465996977, 0.8353631433905236, 0.8349186594363943, 0.835096453018046, 0.8352742465996977, 0.8349186594363943, 0.8345630722730909, 0.8344741754822651, 0.8337630011556583, 0.8337630011556583, 0.832696239665748, 0.8328740332473997, 0.8326073428749222, 0.8323406525024447, 0.8322517557116188, 0.8320739621299671, 0.8318072717574896, 0.8318961685483154, 0.8324295492932705, 0.8326073428749222, 0.8322517557116188, 0.8327851364565739, 0.8311849942217086, 0.831540581385012, 0.8294959551960174, 0.8300293359409725, 0.8286958840785847, 0.8287847808694107, 0.8266512578895902, 0.8271846386345453, 0.8247844252822473, 0.8254955996088541, 0.8222953151391235, 0.8227397990932528, 0.8193617210418704, 0.8197173082051737]
In [105]:
# plot the results (Manhattan distance)
plt.plot(k_potentials, grid_mean_scores)
plt.xlabel('Potential K values for KNN')
plt.ylabel('Cross-Validated Accuracy')
Out[105]:
<matplotlib.text.Text at 0x120cb5a50>

In [106]:
# plot the results (using Euclidean distance)
plt.plot(k_potentials, grid_mean_scores)
plt.xlabel('Potential K values for KNN')
plt.ylabel('Cross-Validated Accuracy')
Out[106]:
<matplotlib.text.Text at 0x12143b7d0>

In [107]:
# examine the best model
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
0.962396657481
{'n_neighbors': 1}
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
metric_params=None, n_jobs=1, n_neighbors=1, p=2,
weights='uniform')
In [ ]:

In [108]:
# read in the data & create matrices
X = X_scaled[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", \
"time_spend_company", "Work_accident", "promotion_last_5years", "low", "medium",\
"technical", "support", "IT", "product_mng", "marketing", "RandD", "accounting",\
"hr"]]
y = X_scaled["left"]
In [176]:
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
Support Vector Machines
In [177]:
# Train, fit and predict with SVM

from sklearn import svm
Model=svm.SVC(kernel='linear')
Model.fit(X_train,y_train)
Y_pred=Model.predict(X_test)
In [178]:
# Cross validation score for SVM
scores = cross_val_score(Model, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.780694712068
In [179]:
# Evaluate SVM for accuracy

from sklearn import metrics
metrics.accuracy_score(y_test,Y_pred)
Out[179]:
0.78106666666666669
In [180]:
# Confusion matrix for SVM

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2696  166]
[ 655  233]]
In [181]:
# Classification Accuracy confirmation between manual calculation and formula for SVM

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.781066666667
0.781066666667
In [182]:
# Misclassification Accuracy confirmation between manual calculation and formula for SVM
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.218933333333
0.218933333333
In [183]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated SVM

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.262387387387
0.262387387387
In [184]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for SVM

print(TN / float(TN + FP))
0.941998602376
In [185]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for SVM

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.583959899749
0.583959899749
In [186]:
# Misclassification Accuracy confirmation between manual calculation and formula for SVM
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.218933333333
0.218933333333
In [187]:
# Roc_curve for Support Vector Machine Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [92]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [93]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Support Vector Machines: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

Random Forest
In [157]:
# Random Forests Classifier from Scikit Learn

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_jobs=2)
RF.fit(X_train,y_train)
Y_pred=RF.predict(X_test)
In [158]:
# Score for 10-fold Cross Validation for Random Forests Classifier

scores = cross_val_score(RF, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.986309602501
In [159]:
# Prediction Accuracy for Random Forests

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.9896
In [160]:
# Confusion matrix for Random Forests

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2857    5]
[  34  854]]
In [161]:
# Classification Accuracy confirmation between manual calculation and formula for Random Forest

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.9896
0.9896
In [162]:
# Misclassification Accuracy confirmation between manual calculation and formula for Random Forest
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0104
0.0104
In [163]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for Random Forest Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.961711711712
0.961711711712
In [164]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for Random Forest Classifier

print(TN / float(TN + FP))
0.998252969951
In [165]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for Random Forest Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.994179278231
0.994179278231
In [166]:
# Misclassification Accuracy confirmation between manual calculation and formula for Random Forest
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0104
0.0104
In [167]:
# Roc_curve for Random Forest Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [168]:
y_pred_prob_Random_Forest = RF.predict_proba(X_test)[:, 1]
In [172]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_Random_Forest, bins=5)
plt.xlim(0, 1)
plt.title('Random Forest Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[172]:
<matplotlib.text.Text at 0x12013a8d0>

In [170]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [171]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Random Forest: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

Bagging
In [109]:
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
In [110]:
# Bagging Classifier

seed = 7
cart = DecisionTreeClassifier()
num_trees = 100
Bagging = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
Bagging.fit(X_train,y_train)
Y_pred=Bagging.predict(X_test)
In [111]:
# Cross Validation for Bagging

scores = cross_val_score(Bagging, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.987910076856
In [112]:
# Prediction Accuracy for Bagging

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.989333333333
In [113]:
# Confusion matrix for Bagging

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2854    8]
[  32  856]]
In [114]:
# Classification Accuracy confirmation between manual calculation and formula for Bagging

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.989333333333
0.989333333333
In [115]:
# Misclassification Accuracy confirmation between manual calculation and formula for Bagging
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0106666666667
0.0106666666667
In [116]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for Bagging Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.963963963964
0.963963963964
In [117]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for Bagging Classifier

print(TN / float(TN + FP))
0.997204751922
In [118]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for Bagging Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.990740740741
0.990740740741
In [119]:
# Misclassification Accuracy confirmation between manual calculation and formula for Bagging
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0106666666667
0.0106666666667
In [120]:
# Roc_curve for Bagging


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [121]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
In [122]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Bagging: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [123]:
y_pred_prob_Bagging = Bagging.predict_proba(X_test)[:, 1]
In [124]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_Bagging, bins=5)
plt.xlim(0, 1)
plt.title('Bagging Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[124]:
<matplotlib.text.Text at 0x11b217610>

In [ ]:

AdaBoost
In [125]:
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
In [126]:
# AdaBoost Classifier

seed = 7
cart = AdaBoostClassifier()
num_trees = 100
AdaBoostClassifier = AdaBoostClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
AdaBoostClassifier.fit(X_train,y_train)
Y_pred=AdaBoostClassifier.predict(X_test)
In [127]:
# Cross Validation for AdaBoost

scores = cross_val_score(AdaBoostClassifier, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.958576016468
In [128]:
# Prediction Accuracy for AdaBoost

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.958133333333
In [129]:
# Confusion matrix for AdaBoost

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2793   69]
[  88  800]]
In [130]:
# Classification Accuracy confirmation between manual calculation and formula for Bagging

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.958133333333
0.958133333333
In [131]:
# Misclassification Accuracy confirmation between manual calculation and formula for AdaBoost
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0418666666667
0.0418666666667
In [132]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for AdaBoost Boosting Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.900900900901
0.900900900901
In [133]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for AdaBoost Classifier

print(TN / float(TN + FP))
0.975890985325
In [134]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for AdaBoost Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.920598388953
0.920598388953
In [135]:
# Misclassification Accuracy confirmation between manual calculation and formula for AdaBoost
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0418666666667
0.0418666666667
In [136]:
# Roc_curve for AdaBoost


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [137]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
In [138]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='AdaBoost: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [139]:
y_pred_prob_AdaBoost = AdaBoostClassifier.predict_proba(X_test)[:, 1]
In [140]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_AdaBoost, bins=5)
plt.xlim(0, 1)
plt.title('AdaBoost Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[140]:
<matplotlib.text.Text at 0x11b367a10>

In [ ]:

In [ ]:

In [ ]:

Gradient Boosting
In [141]:
from sklearn.ensemble import GradientBoostingClassifier
In [142]:
# Train and predict Gradient Boosting

GradientBoosting = GradientBoostingClassifier()
GradientBoosting.fit(X_train,y_train)
Y_pred=GradientBoosting.predict(X_test)
In [143]:
# Cross Validation for Gradient Boosting

scores = cross_val_score(GradientBoosting, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.976443246546
In [144]:
# Prediction Accuracy for Gradient Boosting

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.974133333333
In [145]:
# Confusion matrix for Gradient Boosting

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2832   30]
[  67  821]]
In [146]:
# Classification Accuracy confirmation between manual calculation and formula for Gradient Boosting

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.974133333333
0.974133333333
In [147]:
# Misclassification Accuracy confirmation between manual calculation and formula for Gradient Boosting
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0258666666667
0.0258666666667
In [148]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for Gradient Boosting Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.92454954955
0.92454954955
In [149]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for Gradient Classifier

print(TN / float(TN + FP))
0.989517819706
In [150]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for Gradient Boosting Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.964747356052
0.964747356052
In [151]:
# Misclassification Accuracy confirmation between manual calculation and formula for Gradient Boosting
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0258666666667
0.0258666666667
In [152]:
# Roc_curve for Gradient Boosting


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Gradient Boosting Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [153]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
In [154]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Gradient Boosting: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [155]:
y_pred_prob_GradientBoosting = GradientBoosting.predict_proba(X_test)[:, 1]
In [156]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_GradientBoosting, bins=5)
plt.xlim(0, 1)
plt.title('Gradient Boosting Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[156]:
<matplotlib.text.Text at 0x11c2566d0>

In [ ]:

In [ ]:

Part IV
Conclusion
One interesting chart showed us which departments have the most people leaving (and staying). They are Sales followed by Technical and Support. That means that the biggest effort at the company should be placed on keeping the strong employees in those departments (but by no means neglect the other departments).
There are several other interesting patterns in the data. Some are very intuitive whereas others very surprising and unexpected. There are three clusters of people leaving the company (the most important cluster is not as dense, in the sense that there are actually slightly more people there that stay than leave, but nevertheless its the most interesting and surprising cluster).
The three groups are employees with high performance and low satisfaction, employees with low satisfcation and performance (rather obvious) and employees with high satisfaction and performance (a surprising result). The first two results are very strong but rather trivial, there is limited use in providing detailed analyses of them. However, the somewhat less dense but very significant group of highly satisfied employees that do remarkably well in their evaluations is of extreme importance. Why do they leave? Well, there are several important factors to consider.
The people with high salaries and at even average satisfaction levels rarely leave the firm. Hence, one attribute that people who tend to leave will generally possess is salaries that are not high and fall in the low or medium range. It is interesting to note though that people with near perfect to perfect satisfaction levels do not leave even if they are in the low or medium income range brackets. Hence, a possible course of action to potentially alleviate this issue is increase salaries for good employees. Also, a further investigation into what might be the differentiating factor between people who are very satisfied (0.7-0.9 rate) vs super satisfied (0.9-1.0) can be done to see if it is possible to further increase the satisfaction level of those that are very satisfied but not perfectly so (maybe its possible to do this while at the same time saving money by not instituting as much pay increases if there is a way to get them satisfied in another manner, perhaps by working less hours for instance) would be a good idea.
Secondly, promotions are a huge factor. Looking at the plot of promotions during the past five years, it is very clear that people who are promoted stay and people who are not promoted leave. Of course, promotions and salary increases have some correlation because people tend to receive salary increases during promotions. Hence, it's important to promote employees if they do good work! That will help decrease the turnover rate.
The third important point is that employees who work a lot of hours that stay at the company for at least a medium amount of time tend to leave. Only the relatively small number of employees who work at the company for many years stay at the company regardless of working hours but very few make it there. Hence, the hours worked plays an important decision making role when employees think about whether to stay at the firm or seek other employment. Due to this finding it is important to see if any kind of rebalancing can be done so that part of the work can perhaps be shifted between employees who do not work as many hours and those that do in order to make the distribution of hours worked by employees more uniform which can perhaps increase employees willingness to stay at the company longer. It is quiet possible that some employees just feel burnt out even if they are generally happy with their job. Due to an inefficient work life balance, they might seek to find employment elsewhere.
Multiple machine learning algorithms were ran on the data and quiet a few did relatively well in terms of making predictions on out of sample data as measured by the accuracy score, the confusion matrix and deriatives thereof as well as the ROC Curve. In particular, the Random Forest and Bagging algorithms did the best with accuracy and cross-validation scores marginally different from each other just under 99%. Both correctly predicted over 3,700 out of 3,750 employees accurately (compared to a baseline accuracy of only 76% (2,850) as to whether they would leave/stay). The various scores computed from the confusion matrix were all almost perfect. Particularly importantly, the ROC curve has an area covering of .98 for both those algorithms.
Hence, using either of these two algorithms (Random Forests or Bagging) have given us great predictions as to which employees will leave vs which ones would stay.
Assumptions
We have assumed the dataset is accurate as we have not audited its validity.
Another assumption we made is that we trained on 75% of the data, conducted cross-validation, and then tested on the 25% remaining out of sample. However, when trying to sample to train a larger sample of the data (such as 90/10 test/train split), we received very similar results in our algorithms. Hence, our model is robust as should perform well when used on future data.
An assumption made by the Random Forest Model which did exceptionally well is that the sampling that it conducted during boostrap aggregation is actually representative. This is a safe assumption to make and the model has done really well on out of sample data.
Introduction:
This project analyzes a dataset in an attempt to determine what causes the good employees to leave their current jobs. The first step is to clean the data (Part I) and visualize it (Part II) using various types of graphs and charts to help establish clear relationships between various features. Part III analyzes the dataset using various Machine Learning algorithms and picks the best algorithm to model this dataset based on various metrics. Lastly, a conclusion on the entire analysis is provided.
Part I
Data Cleaning
This initial stage of the project is to clean the data in such a way that all the important attributes of the dataset be retained and unhindered for analysis purposes. Hence, the initial cleaning phase of the project imports the dataset and the libraries needed for the cleaning stages of the project. A check for Null Values/Missing Values is conducted (none are present). The multiple categorical variables in the dataset are created into dummy variables which is necessary for later stages of the analysis. These new variables are afterwards added to the original dataset and the initial variables from which dummification was done are removed to avoid redundancy. Lastly, the dataset is normalized due to the need to apply machine learning algorithms to it in later stages of the project.
In [1]:
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

%matplotlib inline
In [2]:
HR = pd.read_csv('HR_comma_sep.csv')
HR.head(10)
Out[2]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary
0	0.38	0.53	2	157	3	0	1	0	sales	low
1	0.80	0.86	5	262	6	0	1	0	sales	medium
2	0.11	0.88	7	272	4	0	1	0	sales	medium
3	0.72	0.87	5	223	5	0	1	0	sales	low
4	0.37	0.52	2	159	3	0	1	0	sales	low
5	0.41	0.50	2	153	3	0	1	0	sales	low
6	0.10	0.77	6	247	4	0	1	0	sales	low
7	0.92	0.85	5	259	5	0	1	0	sales	low
8	0.89	1.00	5	224	5	0	1	0	sales	low
9	0.42	0.53	2	142	3	0	1	0	sales	low
In [3]:
HR[HR['satisfaction_level'] == np.nan]
HR[HR['last_evaluation'] == np.nan]
HR[HR['number_project'] == np.nan]
HR[HR['average_montly_hours'] == np.nan]
HR[HR['time_spend_company'] == np.nan]
HR[HR['Work_accident'] == np.nan]
HR[HR['left'] == np.nan]
HR[HR['sales'] == np.nan]
HR[HR['salary'] == np.nan]
Out[3]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary
In [4]:
# Create a heatmap, to check for null values.

plt.figure(figsize=(10,7))
sns.heatmap(HR.isnull(),yticklabels=False, cbar=False, cmap='viridis')
Out[4]:
<matplotlib.axes._subplots.AxesSubplot at 0x114c6ee50>

In [5]:
HR['sales'].value_counts()
Out[5]:
sales          4140
technical      2720
support        2229
IT             1227
product_mng     902
marketing       858
RandD           787
accounting      767
hr              739
management      630
Name: sales, dtype: int64
In [6]:
HR['salary'].value_counts()
Out[6]:
low       7316
medium    6446
high      1237
Name: salary, dtype: int64
In [7]:
HR['left'].value_counts()
Out[7]:
0    11428
1     3571
Name: left, dtype: int64
In [8]:
# Create dummy variables for all the departments

sales = pd.get_dummies(HR.sales)

sales.columns = ["IT", "RandD", "accounting", "hr", "management", "marketing", "product_mng", "sales_department", "support", "technical"]

sales.head()
Out[8]:
IT	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical
0	0	0	0	0	0	0	0	1	0	0
1	0	0	0	0	0	0	0	1	0	0
2	0	0	0	0	0	0	0	1	0	0
3	0	0	0	0	0	0	0	1	0	0
4	0	0	0	0	0	0	0	1	0	0
In [9]:
# Create dummy variables for the salary

salary = pd.get_dummies(HR.salary)

salary.columns = ['low', 'medium', 'high']

salary.head()
Out[9]:
low	medium	high
0	0	1	0
1	0	0	1
2	0	0	1
3	0	1	0
4	0	1	0
In [10]:
# Create dummy variable for stayed/left

left = pd.get_dummies(HR.left)

left.columns = ['left', 'stayed']

left.head(10)
Out[10]:
left	stayed
0	0	1
1	0	1
2	0	1
3	0	1
4	0	1
5	0	1
6	0	1
7	0	1
8	0	1
9	0	1
In [11]:
# Concatenate the original dataset with the newly created salary and sales dummified variables

HR = pd.concat([HR, salary, sales], axis = 1)

HR.head(10)
Out[11]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary	...	IT	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical
0	0.38	0.53	2	157	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
1	0.80	0.86	5	262	6	0	1	0	sales	medium	...	0	0	0	0	0	0	0	1	0	0
2	0.11	0.88	7	272	4	0	1	0	sales	medium	...	0	0	0	0	0	0	0	1	0	0
3	0.72	0.87	5	223	5	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
4	0.37	0.52	2	159	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
5	0.41	0.50	2	153	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
6	0.10	0.77	6	247	4	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
7	0.92	0.85	5	259	5	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
8	0.89	1.00	5	224	5	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
9	0.42	0.53	2	142	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
10 rows × 23 columns
In [12]:
HR.dtypes
Out[12]:
satisfaction_level       float64
last_evaluation          float64
number_project             int64
average_montly_hours       int64
time_spend_company         int64
Work_accident              int64
left                       int64
promotion_last_5years      int64
sales                     object
salary                    object
low                        uint8
medium                     uint8
high                       uint8
IT                         uint8
RandD                      uint8
accounting                 uint8
hr                         uint8
management                 uint8
marketing                  uint8
product_mng                uint8
sales_department           uint8
support                    uint8
technical                  uint8
dtype: object
In [13]:
# Create a new column 

salary_map = {'low': 1, 'medium': 2, 'high': 3}

HR['salary_variable'] = HR['salary'].apply(lambda x: salary_map[x])

HR.head()
Out[13]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary	...	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical	salary_variable
0	0.38	0.53	2	157	3	0	1	0	sales	low	...	0	0	0	0	0	0	1	0	0	1
1	0.80	0.86	5	262	6	0	1	0	sales	medium	...	0	0	0	0	0	0	1	0	0	2
2	0.11	0.88	7	272	4	0	1	0	sales	medium	...	0	0	0	0	0	0	1	0	0	2
3	0.72	0.87	5	223	5	0	1	0	sales	low	...	0	0	0	0	0	0	1	0	0	1
4	0.37	0.52	2	159	3	0	1	0	sales	low	...	0	0	0	0	0	0	1	0	0	1
5 rows × 24 columns
In [14]:
del HR['salary']

del HR['sales']
In [15]:
HR.head()
Out[15]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	low	medium	...	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical	salary_variable
0	0.38	0.53	2	157	3	0	1	0	0	1	...	0	0	0	0	0	0	1	0	0	1
1	0.80	0.86	5	262	6	0	1	0	0	0	...	0	0	0	0	0	0	1	0	0	2
2	0.11	0.88	7	272	4	0	1	0	0	0	...	0	0	0	0	0	0	1	0	0	2
3	0.72	0.87	5	223	5	0	1	0	0	1	...	0	0	0	0	0	0	1	0	0	1
4	0.37	0.52	2	159	3	0	1	0	0	1	...	0	0	0	0	0	0	1	0	0	1
5 rows × 22 columns
In [16]:
HR.dtypes
Out[16]:
satisfaction_level       float64
last_evaluation          float64
number_project             int64
average_montly_hours       int64
time_spend_company         int64
Work_accident              int64
left                       int64
promotion_last_5years      int64
low                        uint8
medium                     uint8
high                       uint8
IT                         uint8
RandD                      uint8
accounting                 uint8
hr                         uint8
management                 uint8
marketing                  uint8
product_mng                uint8
sales_department           uint8
support                    uint8
technical                  uint8
salary_variable            int64
dtype: object
In [17]:
from sklearn.preprocessing import MinMaxScaler
X_scaled = MinMaxScaler().fit_transform(HR.astype(float))
X_scaled = pd.DataFrame(X_scaled, columns = HR.columns)

# Min Max Scaler

X_scaled.head(10)
Out[17]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	low	medium	...	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical	salary_variable
0	0.318681	0.265625	0.0	0.285047	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
1	0.780220	0.781250	0.6	0.775701	0.500	0.0	1.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.5
2	0.021978	0.812500	1.0	0.822430	0.250	0.0	1.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.5
3	0.692308	0.796875	0.6	0.593458	0.375	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
4	0.307692	0.250000	0.0	0.294393	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
5	0.351648	0.218750	0.0	0.266355	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
6	0.010989	0.640625	0.8	0.705607	0.250	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
7	0.912088	0.765625	0.6	0.761682	0.375	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
8	0.879121	1.000000	0.6	0.598131	0.375	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
9	0.362637	0.265625	0.0	0.214953	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
10 rows × 22 columns
Part II
Exploratory Data Analysis (EDA)
In this section, we try to visualize the data on employees using various plots, graphs, charts in order to get an idea of how different features in the dataset relate to each other.
The below represents a heatmap of correlations of features. Some interesting observations from this heatmap: Quiet intuitively, you can see that the relationship between satisfaction level and people that left the company is negative. It's also interesting to note that there is a positive correlation between the number of hours employees work and their evaluation. These and other relationships will be explored more deeply in the rest of the EDA.
In [18]:
correlation = X_scaled.corr()
plt.figure(figsize=(20,20))
sns.heatmap(correlation, vmax=10, square=True,annot=True,cmap=None, linewidths=1)

plt.title('Correlation between features')
Out[18]:
<matplotlib.text.Text at 0x118f3a050>

In [19]:
# Exported csv file to work with in Tableau
X_scaled.to_csv("Updated_Capstone.csv")
In [20]:
X_scaled['left'].value_counts()
Out[20]:
0.0    11428
1.0     3571
Name: left, dtype: int64
In [21]:
stayed = np.where(X_scaled['left'] == 0)[0]
left = np.where(X_scaled['left'] == 1)[0]
For a given level of satisfaction, there is definitely an intuitive pattern that the higher someone's salary is, the less they are likely to leave. This relationship is a lot more apparent in the distinction between the high income earners vs low & median income earners as a group. The difference in the people that left between low and median income salaries is not significant. Hence, most of the people that leave are not receiving a high salary, which means that you if you would like to retain these employees, you should increase their pay!
In [22]:
# plot of Satisfaction vs Last Evaluation
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.satisfaction_level[stayed], X_scaled.salary_variable[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.satisfaction_level[left], X_scaled.salary_variable[left], alpha=0.1, color='r')
ax.set_title("satisfaction_level vs salary")
ax.set_xlabel("satisfaction_level")
ax.set_ylabel("salary")
plt.show()

In the Satisfaction vs Last Evaluation plot below, we can note some interesting observations. Firstly, there is a sizeable cluster of people with strong evaluations and low satisfaction levels leaving. Secondly, there is a significant group of people leaving that are slighly below average in performance and evaluation. Lastly, and by far the most interesting observation of the three, there is a decent size of a not very dense cluster of people leaving that have done extremely well on their evaluations (0.7 to 1 on a 0-1 scale) and and have very high satisfaction rates, albeit not perfect (0.7-0.9 on a 0-1 scale). That particular group of employees also have many people that stay, but a significant proportion of that group is leaving. So the most surprising thing to observe is that people can be satisfied with the job and still leave! There are several possibilites for why this might be possible. People might enjoy their job but think they can do even better elsewhere, whether its income or career growth (or a combination of the two, considering that those things tend to go hand in hand). Another possibility is that people might not be honest in surveys because of fear of being reprimanded for negative surveys.
In [23]:
# plot of Satisfaction vs Last Evaluation
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.satisfaction_level[stayed], X_scaled.last_evaluation[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.satisfaction_level[left], X_scaled.last_evaluation[left], alpha=0.1, color='r')
ax.set_title("satisfaction_level vs last_evaluation")
ax.set_xlabel("satisfaction_level")
ax.set_ylabel("last_evaluation")
plt.show()

The Satisfaction vs Promotion during the last five years plot below shows a very strong relationship in several interesting ways! People who have been promoted don't leave very often! However, such promotions are not common enough! Therefore, if you don't want good employees to leave, promoting them will really help!
In [24]:
# plot of Satisfaction vs Promotion during last five years
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.satisfaction_level[stayed], X_scaled.promotion_last_5years[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.satisfaction_level[left], X_scaled.promotion_last_5years[left], alpha=0.1, color='r')
ax.set_title("satisfaction_level vs promotion_last_5years")
ax.set_xlabel("satisfaction_level")
ax.set_ylabel("promotion_last_5years")
plt.show()

The below scatter plot compares the average working hours against the time spent at the company and how it impacts the employee turnover rate. There are multiple interesting relationships to note. For employees that just started out, the majority tend to stick around for some time even if they work substantial hours. However, for employees who have been at the company for a while, there is a significant tendency to leave the company if working substantial hours. Consequently, we can observe that very few employees stick around at the company for a long time period because the ones who were working long hours almost all left! However, those that do stick around for a long time (relatively few) tend to all be devoted and not leave
There are several conjectures for why this is possible. One is that employees have worked for a significant amount of hours and put in a lot of time and effort expecting a quick promotion and/or pay increase (refer to graph above for relationship between promotions and staying/leaving) that they have not received and leave for elsewhere. Another reason might be that people have tried working for a little bit of time, realized that they are being overworked and have started looking for other opportunities to seek employment.
In [25]:
# plot of Average_monthly_hours vs time_spend_company
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.average_montly_hours[stayed], X_scaled.time_spend_company[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.average_montly_hours[left], X_scaled.time_spend_company[left], alpha=0.1, color='r')
ax.set_title("average_monthly_hours vs time_spend_company")
ax.set_xlabel("average_monthly_hours")
ax.set_ylabel("time_spend_company")
plt.show()

In [26]:
# Imported from Tableau
from IPython.display import Image
Image(filename = 'Average Monthly Hours vs Satisfaction Level.png', width = 1000)
Out[26]:

There does not appear to be a strong relationship between accidents and people leaving the company, hence, people stay/leave regardless of the accidents they might have experienced.
In [27]:
# plot of work_accident vs time_spend_company
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.Work_accident[stayed], X_scaled.time_spend_company[stayed], alpha=0.5, color='g')
ax.scatter(X_scaled.Work_accident[left], X_scaled.time_spend_company[left], alpha=0.5, color='r')
ax.set_title("Work_accident vs time_spend_company")
ax.set_xlabel("Work_accident")
ax.set_ylabel("time_spend_company")
plt.show()

The below stripplot compares number of projects done by an employee versus the number of promotions they have received over the past 5 years. It is interesting to note that the number of projects completed has very insignificant impact on whether a person leaves. On the contrary, the promotions received during the past 5 years is of critical importance. The majority of people who have received a promotion stay. Almost all those who were not promoted leave.
The data seems insufficient to be able to determine why the number of projects is not related to promotions. There are at least several possibilities. Perhaps the number of projects refers to the number assigned (vs completed) or that the projects are not all of equal difficulty and not equally time consuming leading to distorted hypothesis that the number of projects might necessarily correspond to an employee's productivity.
In [28]:
import seaborn as sns

sns.stripplot("number_project", "promotion_last_5years", data=X_scaled, hue="left", jitter = True)
Out[28]:
<matplotlib.axes._subplots.AxesSubplot at 0x117949d90>

The Satisfaction level distribution below shows that the majority of people are at least moderately satisfied with their job. Therefore, for the people that are satisfied and do well, efforts should be placed on making them even more happy by promoting them. For the people that do poorly or mediocre, either studies should be placed on determining why they are not doing well and trying to improve their performance or attempting to replace them with better employees (this would depend on a cost/benefit analysis) done by the company regarding this issue.
In [29]:
# Histogram distribution of Satisfaction Level

f,ax1 = plt.subplots(1,1)
sns.distplot(X_scaled['satisfaction_level'],bins=10,norm_hist=False)
plt.show()

The distribution (which is a decent approximation of a normal distribution with almost no tails and some kurtosis present) of the last evaluation graph shows us there is a sizeable portion of people that do well in their evaluations, hence the company should try to do as much as possible to keep those people!
In [30]:
# Histogram distribution of Last Evaluation


f,ax1 = plt.subplots(1,1)
sns.distplot(X_scaled['last_evaluation'],bins=10,norm_hist=False)
plt.show()

The below plot shows the ratios of people who stay in the company by department and the ratio of people who leave the company by department. It is clear from the chart that the departments with the most number of employees also have the highest ratios of both people who have left and those that stayed at the company. This tells us there should be particular effort place on retaining good employees that work in those departments, in particular the sales, technical and support departments, respectively. The largest efforts should be placed on retaining the good employees in these groups because they have the most people leaving.
In [34]:
# This plot is to compare the people that left across different departments
# The non-department variables are deleted to only keep the departments

del X_scaled['satisfaction_level']
del X_scaled['last_evaluation']
del X_scaled['number_project']
del X_scaled['average_montly_hours']
del X_scaled['time_spend_company']
del X_scaled['Work_accident']
del X_scaled['promotion_last_5years']
del X_scaled['low']
del X_scaled['medium']
del X_scaled['high']
del X_scaled['salary_variable']


left=X_scaled.groupby('left').mean()
left=left.transpose()
plt.figure(figsize=(20,10))
left.plot(kind='bar')
Out[34]:
<matplotlib.axes._subplots.AxesSubplot at 0x11a7bcd10>
<matplotlib.figure.Figure at 0x11a360850>

Part III
Predictive Modelling
In this section below, multiple models will be run that are applicable to classification. Afterwards, various scoring metrics will be calculated for the models and a determination will be made as to which model is based for this dataset.
In [174]:
# read in the data & create matrices
X = X_scaled[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", \
"time_spend_company", "Work_accident", "promotion_last_5years", "salary_variable",\
"technical", "support", "IT", "product_mng", "marketing", "RandD", "accounting",\
"hr", "management", "sales_department"]]
y = X_scaled["left"]
In [175]:
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
Logistic Regression Model
In [33]:
# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
model = LogisticRegression()

# fit the model with data
mdl = model.fit(X_train, y_train)

# predict the response for new observations

logit = model.predict(X_test)

# sklearn output to check that ran on right data split.

len(logit)
Out[33]:
3750
In [34]:
# These represent the y-intercept and coefficients for all the variables in the logistic regression.

print(model.intercept_)
print(model.coef_)
[ 0.57305398]
[[-3.71487625  0.45484967 -1.50619896  0.8704181   1.97590531 -1.45425383
-1.26024249 -1.32512177  0.22588571  0.23497924 -0.01282827  0.07509077
0.13925062 -0.47542127  0.26606552  0.3920348  -0.44451555  0.17251241]]
In [35]:
# Prediction Accuracy for Logistic Regression

from sklearn import metrics
print(metrics.accuracy_score(y_test, logit))
0.796
In [36]:
from sklearn.cross_validation import cross_val_score
In [37]:
# 10-fold cross-validation for Logistic Regression to find the CV score
mdl = model.fit(X_train, y_train)
scores = cross_val_score(mdl, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.794205514996
In [38]:
from sklearn import metrics
confusion = metrics.confusion_matrix(y_test, logit)
print ((confusion))
[[2651  211]
[ 554  334]]
In [39]:
y_pred_prob = model.predict_proba(X_test)[:, 1]
In [40]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob, bins=5)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[40]:
<matplotlib.text.Text at 0x1181a6a90>

In [41]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [42]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, logit)


plot_confusion_matrix(cm, title='LogisticRegression: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [43]:
# Roc_curve for Logistic Regression Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, logit)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Logistic Regression Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

K Nearest Nearbours Model
K=1
In [44]:
from sklearn.neighbors import KNeighborsClassifier
In [45]:
# K-Nearest Neighbours with K=1
knn = KNeighborsClassifier(n_neighbors=1, metric = 'euclidean')
In [46]:
y = knn.fit(X_train, y_train)

KNN = knn.predict(X_test)

# sklearn output to check that ran on right data split.

len(KNN)
Out[46]:
3750
In [47]:
# Cross fold validation for K=1

md2 = knn.fit(X_train, y_train)
scores = cross_val_score(md2, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.962043001151
In [48]:
# Prediction Accuracy.

from sklearn import metrics
print(metrics.accuracy_score(y_test, KNN))
0.967466666667
In [49]:
confusion = metrics.confusion_matrix(y_test, KNN)
print confusion
[[2775   87]
[  35  853]]
In [50]:
# Classification Accuracy confirmation between manual calculation and formula for K=1

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, KNN))
0.967466666667
0.967466666667
In [51]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=1
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN))
0.0325333333333
0.0325333333333
In [52]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for K=1

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, KNN))
0.960585585586
0.960585585586
In [53]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for K=1


print(TN / float(TN + FP))
0.969601677149
In [54]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for K=1

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, KNN))
0.907446808511
0.907446808511
In [55]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=1
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN))
0.0325333333333
0.0325333333333
In [56]:
y_pred_prob_knn = knn.predict_proba(X_test)[:, 1]
In [57]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_knn, bins=5)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[57]:
<matplotlib.text.Text at 0x1181d6b90>

In [58]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [59]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, KNN)


plot_confusion_matrix(cm, title='KNN for K=1: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [60]:
# Roc_curve for KNN=1 Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, KNN)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('KNN=1 Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

K=5
In [61]:
# K-Nearest Neighbours with K=5

knn = KNeighborsClassifier(n_neighbors=5, metric = 'euclidean')
y = knn.fit(X_train, y_train)
KNN5 = knn.predict(X_test)

# sklearn output to check that ran on right data split.

len(KNN5)
Out[61]:
3750
In [62]:
# Cross Validation for K=5 Cross-Fold Validation

md2 = knn.fit(X_train, y_train)
scores = cross_val_score(md2, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.938309179837
In [63]:
# Prediction Accuracy on KNN=5 for accuracy score

from sklearn import metrics
print(metrics.accuracy_score(y_test, KNN5))
0.941333333333
In [64]:
confusion = metrics.confusion_matrix(y_test, KNN5)
print confusion
[[2731  131]
[  89  799]]
In [65]:
# Classification Accuracy confirmation between manual calculation and formula for K=5

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, KNN5))
0.941333333333
0.941333333333
In [66]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=5
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN5))
0.0586666666667
0.0586666666667
In [67]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for K=5

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, KNN5))
0.899774774775
0.899774774775
In [68]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for K=5

print(TN / float(TN + FP))
0.954227812718
In [69]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for K=5

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, KNN5))
0.859139784946
0.859139784946
In [70]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=5
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN5))
0.0586666666667
0.0586666666667
In [71]:
# Roc_curve for KNN=5 Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, KNN5)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [72]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [73]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, KNN5)


plot_confusion_matrix(cm, title='KNN for N=5: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [74]:
y_pred_prob_knn = knn.predict_proba(X_test)[:, 1]
In [75]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_knn, bins=5)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[75]:
<matplotlib.text.Text at 0x11acdea10>

In [ ]:

Grid Search for optimal N value in KNN
In [101]:
from sklearn.grid_search import GridSearchCV
/Users/Misha/anaconda/lib/python2.7/site-packages/sklearn/grid_search.py:43: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.
DeprecationWarning)
In [102]:
k_potentials = list(range(1, 200))
print(k_potentials)
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]
In [103]:
# Set Nearest Neighbours equal to k
knn = KNeighborsClassifier(n_neighbors=k_potentials, metric = 'euclidean')
# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_potentials)
# instantiate the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
# fit the grid with data
grid.fit(X_train, y_train)
# view the results
grid.grid_scores_
Out[103]:
[mean: 0.96240, std: 0.00591, params: {'n_neighbors': 1},
mean: 0.95884, std: 0.00543, params: {'n_neighbors': 2},
mean: 0.94568, std: 0.00463, params: {'n_neighbors': 3},
mean: 0.94622, std: 0.00555, params: {'n_neighbors': 4},
mean: 0.93875, std: 0.00620, params: {'n_neighbors': 5},
mean: 0.94284, std: 0.00673, params: {'n_neighbors': 6},
mean: 0.93795, std: 0.00746, params: {'n_neighbors': 7},
mean: 0.93839, std: 0.00819, params: {'n_neighbors': 8},
mean: 0.93351, std: 0.00932, params: {'n_neighbors': 9},
mean: 0.93493, std: 0.00848, params: {'n_neighbors': 10},
mean: 0.93048, std: 0.00825, params: {'n_neighbors': 11},
mean: 0.93253, std: 0.00810, params: {'n_neighbors': 12},
mean: 0.92933, std: 0.00898, params: {'n_neighbors': 13},
mean: 0.92968, std: 0.00824, params: {'n_neighbors': 14},
mean: 0.92719, std: 0.00747, params: {'n_neighbors': 15},
mean: 0.92817, std: 0.00830, params: {'n_neighbors': 16},
mean: 0.92559, std: 0.00873, params: {'n_neighbors': 17},
mean: 0.92728, std: 0.00821, params: {'n_neighbors': 18},
mean: 0.92497, std: 0.00821, params: {'n_neighbors': 19},
mean: 0.92470, std: 0.00830, params: {'n_neighbors': 20},
mean: 0.92204, std: 0.00804, params: {'n_neighbors': 21},
mean: 0.92186, std: 0.00867, params: {'n_neighbors': 22},
mean: 0.91902, std: 0.00856, params: {'n_neighbors': 23},
mean: 0.91777, std: 0.00850, params: {'n_neighbors': 24},
mean: 0.91564, std: 0.00809, params: {'n_neighbors': 25},
mean: 0.91457, std: 0.00913, params: {'n_neighbors': 26},
mean: 0.91244, std: 0.00927, params: {'n_neighbors': 27},
mean: 0.91244, std: 0.00873, params: {'n_neighbors': 28},
mean: 0.91101, std: 0.00878, params: {'n_neighbors': 29},
mean: 0.91110, std: 0.00853, params: {'n_neighbors': 30},
mean: 0.90906, std: 0.00855, params: {'n_neighbors': 31},
mean: 0.90941, std: 0.00906, params: {'n_neighbors': 32},
mean: 0.90773, std: 0.00922, params: {'n_neighbors': 33},
mean: 0.90728, std: 0.00991, params: {'n_neighbors': 34},
mean: 0.90550, std: 0.00946, params: {'n_neighbors': 35},
mean: 0.90417, std: 0.00891, params: {'n_neighbors': 36},
mean: 0.90292, std: 0.00887, params: {'n_neighbors': 37},
mean: 0.90097, std: 0.00881, params: {'n_neighbors': 38},
mean: 0.90035, std: 0.00948, params: {'n_neighbors': 39},
mean: 0.89875, std: 0.01009, params: {'n_neighbors': 40},
mean: 0.89759, std: 0.01009, params: {'n_neighbors': 41},
mean: 0.89626, std: 0.00955, params: {'n_neighbors': 42},
mean: 0.89537, std: 0.01016, params: {'n_neighbors': 43},
mean: 0.89270, std: 0.01221, params: {'n_neighbors': 44},
mean: 0.89172, std: 0.01244, params: {'n_neighbors': 45},
mean: 0.88950, std: 0.01089, params: {'n_neighbors': 46},
mean: 0.88906, std: 0.01150, params: {'n_neighbors': 47},
mean: 0.88666, std: 0.01137, params: {'n_neighbors': 48},
mean: 0.88577, std: 0.01143, params: {'n_neighbors': 49},
mean: 0.88523, std: 0.01132, params: {'n_neighbors': 50},
mean: 0.88452, std: 0.01149, params: {'n_neighbors': 51},
mean: 0.88372, std: 0.01237, params: {'n_neighbors': 52},
mean: 0.88283, std: 0.01227, params: {'n_neighbors': 53},
mean: 0.88141, std: 0.01166, params: {'n_neighbors': 54},
mean: 0.88114, std: 0.01085, params: {'n_neighbors': 55},
mean: 0.88061, std: 0.01041, params: {'n_neighbors': 56},
mean: 0.87990, std: 0.01105, params: {'n_neighbors': 57},
mean: 0.87910, std: 0.00974, params: {'n_neighbors': 58},
mean: 0.87892, std: 0.00997, params: {'n_neighbors': 59},
mean: 0.87803, std: 0.01108, params: {'n_neighbors': 60},
mean: 0.87768, std: 0.01119, params: {'n_neighbors': 61},
mean: 0.87688, std: 0.01152, params: {'n_neighbors': 62},
mean: 0.87652, std: 0.01182, params: {'n_neighbors': 63},
mean: 0.87590, std: 0.01129, params: {'n_neighbors': 64},
mean: 0.87554, std: 0.01148, params: {'n_neighbors': 65},
mean: 0.87474, std: 0.01183, params: {'n_neighbors': 66},
mean: 0.87412, std: 0.01223, params: {'n_neighbors': 67},
mean: 0.87394, std: 0.01196, params: {'n_neighbors': 68},
mean: 0.87341, std: 0.01210, params: {'n_neighbors': 69},
mean: 0.87314, std: 0.01231, params: {'n_neighbors': 70},
mean: 0.87270, std: 0.01219, params: {'n_neighbors': 71},
mean: 0.87226, std: 0.01168, params: {'n_neighbors': 72},
mean: 0.87154, std: 0.01185, params: {'n_neighbors': 73},
mean: 0.87074, std: 0.01104, params: {'n_neighbors': 74},
mean: 0.87057, std: 0.01132, params: {'n_neighbors': 75},
mean: 0.87003, std: 0.01129, params: {'n_neighbors': 76},
mean: 0.86968, std: 0.01148, params: {'n_neighbors': 77},
mean: 0.86754, std: 0.01056, params: {'n_neighbors': 78},
mean: 0.86692, std: 0.01041, params: {'n_neighbors': 79},
mean: 0.86568, std: 0.01028, params: {'n_neighbors': 80},
mean: 0.86594, std: 0.01024, params: {'n_neighbors': 81},
mean: 0.86461, std: 0.00937, params: {'n_neighbors': 82},
mean: 0.86390, std: 0.00915, params: {'n_neighbors': 83},
mean: 0.86239, std: 0.00956, params: {'n_neighbors': 84},
mean: 0.86257, std: 0.01001, params: {'n_neighbors': 85},
mean: 0.86088, std: 0.01023, params: {'n_neighbors': 86},
mean: 0.86088, std: 0.01051, params: {'n_neighbors': 87},
mean: 0.85981, std: 0.01214, params: {'n_neighbors': 88},
mean: 0.85937, std: 0.01208, params: {'n_neighbors': 89},
mean: 0.85830, std: 0.01087, params: {'n_neighbors': 90},
mean: 0.85821, std: 0.01108, params: {'n_neighbors': 91},
mean: 0.85616, std: 0.00995, params: {'n_neighbors': 92},
mean: 0.85652, std: 0.00954, params: {'n_neighbors': 93},
mean: 0.85519, std: 0.00892, params: {'n_neighbors': 94},
mean: 0.85545, std: 0.00957, params: {'n_neighbors': 95},
mean: 0.85368, std: 0.00900, params: {'n_neighbors': 96},
mean: 0.85332, std: 0.00845, params: {'n_neighbors': 97},
mean: 0.85199, std: 0.00923, params: {'n_neighbors': 98},
mean: 0.85190, std: 0.00926, params: {'n_neighbors': 99},
mean: 0.85021, std: 0.00825, params: {'n_neighbors': 100},
mean: 0.85039, std: 0.00930, params: {'n_neighbors': 101},
mean: 0.84834, std: 0.01015, params: {'n_neighbors': 102},
mean: 0.84825, std: 0.01016, params: {'n_neighbors': 103},
mean: 0.84559, std: 0.00954, params: {'n_neighbors': 104},
mean: 0.84559, std: 0.00922, params: {'n_neighbors': 105},
mean: 0.84416, std: 0.00977, params: {'n_neighbors': 106},
mean: 0.84452, std: 0.00978, params: {'n_neighbors': 107},
mean: 0.84372, std: 0.00935, params: {'n_neighbors': 108},
mean: 0.84399, std: 0.00922, params: {'n_neighbors': 109},
mean: 0.84256, std: 0.00943, params: {'n_neighbors': 110},
mean: 0.84239, std: 0.00930, params: {'n_neighbors': 111},
mean: 0.84132, std: 0.00941, params: {'n_neighbors': 112},
mean: 0.84087, std: 0.00924, params: {'n_neighbors': 113},
mean: 0.84034, std: 0.00864, params: {'n_neighbors': 114},
mean: 0.84016, std: 0.00875, params: {'n_neighbors': 115},
mean: 0.83981, std: 0.00849, params: {'n_neighbors': 116},
mean: 0.83972, std: 0.00862, params: {'n_neighbors': 117},
mean: 0.83865, std: 0.00779, params: {'n_neighbors': 118},
mean: 0.83892, std: 0.00782, params: {'n_neighbors': 119},
mean: 0.83883, std: 0.00798, params: {'n_neighbors': 120},
mean: 0.83892, std: 0.00810, params: {'n_neighbors': 121},
mean: 0.83661, std: 0.00577, params: {'n_neighbors': 122},
mean: 0.83705, std: 0.00582, params: {'n_neighbors': 123},
mean: 0.83670, std: 0.00582, params: {'n_neighbors': 124},
mean: 0.83714, std: 0.00545, params: {'n_neighbors': 125},
mean: 0.83732, std: 0.00565, params: {'n_neighbors': 126},
mean: 0.83741, std: 0.00610, params: {'n_neighbors': 127},
mean: 0.83732, std: 0.00647, params: {'n_neighbors': 128},
mean: 0.83741, std: 0.00634, params: {'n_neighbors': 129},
mean: 0.83759, std: 0.00626, params: {'n_neighbors': 130},
mean: 0.83750, std: 0.00609, params: {'n_neighbors': 131},
mean: 0.83759, std: 0.00639, params: {'n_neighbors': 132},
mean: 0.83767, std: 0.00616, params: {'n_neighbors': 133},
mean: 0.83732, std: 0.00640, params: {'n_neighbors': 134},
mean: 0.83776, std: 0.00641, params: {'n_neighbors': 135},
mean: 0.83723, std: 0.00690, params: {'n_neighbors': 136},
mean: 0.83794, std: 0.00675, params: {'n_neighbors': 137},
mean: 0.83687, std: 0.00710, params: {'n_neighbors': 138},
mean: 0.83687, std: 0.00718, params: {'n_neighbors': 139},
mean: 0.83607, std: 0.00744, params: {'n_neighbors': 140},
mean: 0.83625, std: 0.00725, params: {'n_neighbors': 141},
mean: 0.83501, std: 0.00739, params: {'n_neighbors': 142},
mean: 0.83536, std: 0.00761, params: {'n_neighbors': 143},
mean: 0.83430, std: 0.00774, params: {'n_neighbors': 144},
mean: 0.83536, std: 0.00776, params: {'n_neighbors': 145},
mean: 0.83510, std: 0.00746, params: {'n_neighbors': 146},
mean: 0.83527, std: 0.00732, params: {'n_neighbors': 147},
mean: 0.83412, std: 0.00716, params: {'n_neighbors': 148},
mean: 0.83492, std: 0.00704, params: {'n_neighbors': 149},
mean: 0.83474, std: 0.00740, params: {'n_neighbors': 150},
mean: 0.83483, std: 0.00759, params: {'n_neighbors': 151},
mean: 0.83492, std: 0.00751, params: {'n_neighbors': 152},
mean: 0.83536, std: 0.00788, params: {'n_neighbors': 153},
mean: 0.83492, std: 0.00833, params: {'n_neighbors': 154},
mean: 0.83492, std: 0.00841, params: {'n_neighbors': 155},
mean: 0.83492, std: 0.00815, params: {'n_neighbors': 156},
mean: 0.83510, std: 0.00822, params: {'n_neighbors': 157},
mean: 0.83527, std: 0.00826, params: {'n_neighbors': 158},
mean: 0.83554, std: 0.00832, params: {'n_neighbors': 159},
mean: 0.83554, std: 0.00778, params: {'n_neighbors': 160},
mean: 0.83519, std: 0.00745, params: {'n_neighbors': 161},
mean: 0.83519, std: 0.00757, params: {'n_neighbors': 162},
mean: 0.83536, std: 0.00715, params: {'n_neighbors': 163},
mean: 0.83527, std: 0.00767, params: {'n_neighbors': 164},
mean: 0.83536, std: 0.00778, params: {'n_neighbors': 165},
mean: 0.83492, std: 0.00759, params: {'n_neighbors': 166},
mean: 0.83510, std: 0.00790, params: {'n_neighbors': 167},
mean: 0.83527, std: 0.00820, params: {'n_neighbors': 168},
mean: 0.83492, std: 0.00775, params: {'n_neighbors': 169},
mean: 0.83456, std: 0.00746, params: {'n_neighbors': 170},
mean: 0.83447, std: 0.00782, params: {'n_neighbors': 171},
mean: 0.83376, std: 0.00770, params: {'n_neighbors': 172},
mean: 0.83376, std: 0.00731, params: {'n_neighbors': 173},
mean: 0.83270, std: 0.00737, params: {'n_neighbors': 174},
mean: 0.83287, std: 0.00736, params: {'n_neighbors': 175},
mean: 0.83261, std: 0.00701, params: {'n_neighbors': 176},
mean: 0.83234, std: 0.00676, params: {'n_neighbors': 177},
mean: 0.83225, std: 0.00719, params: {'n_neighbors': 178},
mean: 0.83207, std: 0.00718, params: {'n_neighbors': 179},
mean: 0.83181, std: 0.00746, params: {'n_neighbors': 180},
mean: 0.83190, std: 0.00741, params: {'n_neighbors': 181},
mean: 0.83243, std: 0.00808, params: {'n_neighbors': 182},
mean: 0.83261, std: 0.00827, params: {'n_neighbors': 183},
mean: 0.83225, std: 0.00819, params: {'n_neighbors': 184},
mean: 0.83279, std: 0.00779, params: {'n_neighbors': 185},
mean: 0.83118, std: 0.00688, params: {'n_neighbors': 186},
mean: 0.83154, std: 0.00684, params: {'n_neighbors': 187},
mean: 0.82950, std: 0.00716, params: {'n_neighbors': 188},
mean: 0.83003, std: 0.00689, params: {'n_neighbors': 189},
mean: 0.82870, std: 0.00649, params: {'n_neighbors': 190},
mean: 0.82878, std: 0.00704, params: {'n_neighbors': 191},
mean: 0.82665, std: 0.00624, params: {'n_neighbors': 192},
mean: 0.82718, std: 0.00654, params: {'n_neighbors': 193},
mean: 0.82478, std: 0.00895, params: {'n_neighbors': 194},
mean: 0.82550, std: 0.00846, params: {'n_neighbors': 195},
mean: 0.82230, std: 0.00974, params: {'n_neighbors': 196},
mean: 0.82274, std: 0.01041, params: {'n_neighbors': 197},
mean: 0.81936, std: 0.00926, params: {'n_neighbors': 198},
mean: 0.81972, std: 0.00963, params: {'n_neighbors': 199}]
In [104]:
# create a list of the mean scores only
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print(grid_mean_scores)
[0.9623966574806649, 0.9588407858476309, 0.9456840608054049, 0.94621744155036, 0.9387501111209885, 0.9428393634989777, 0.9379500400035559, 0.9383945239576851, 0.9335052004622633, 0.9349275491154769, 0.9304827095741843, 0.9325273357631789, 0.9293270512934483, 0.9296826384567517, 0.9271935283136279, 0.9281713930127122, 0.9255933860787625, 0.9272824251044537, 0.9249711085429816, 0.9247044181705041, 0.9220375144457285, 0.9218597208640769, 0.9190150235576495, 0.9177704684860877, 0.9156369455062672, 0.914570184016357, 0.9124366610365365, 0.9124366610365365, 0.911014312383323, 0.9111032091741488, 0.9090585829851542, 0.9094141701484576, 0.9077251311227664, 0.9072806471686372, 0.9055027113521202, 0.9041692594897324, 0.9029247044181705, 0.9009689750200017, 0.9003466974842208, 0.8987465552493555, 0.8975908969686194, 0.8962574451062316, 0.8953684771979732, 0.8927015734731976, 0.8917237087741132, 0.8895012890034669, 0.8890568050493377, 0.8866565916970397, 0.8857676237887813, 0.8852342430438261, 0.8845230687172193, 0.8837229975997867, 0.8828340296915281, 0.8814116810383145, 0.881144990665837, 0.8806116099208818, 0.879900435594275, 0.8791003644768424, 0.8789225708951907, 0.8780336029869322, 0.8776780158236288, 0.8768779447061961, 0.8765223575428926, 0.8759000800071117, 0.8755444928438083, 0.8747444217263757, 0.8741221441905948, 0.873944350608943, 0.8734109698639879, 0.8731442794915104, 0.8726997955373811, 0.8722553115832519, 0.871544137256645, 0.8707440661392124, 0.8705662725575607, 0.8700328918126056, 0.8696773046493022, 0.8675437816694818, 0.8669215041337007, 0.8656769490621389, 0.8659436394346164, 0.8646101875722286, 0.8638990132456218, 0.8623877678015823, 0.862565561383234, 0.8608765223575429, 0.8608765223575429, 0.8598097608676327, 0.8593652769135034, 0.8582985154235933, 0.8582096186327673, 0.8561649924437728, 0.8565205796070762, 0.8551871277446884, 0.8554538181171659, 0.853675882300649, 0.8533202951373455, 0.8519868432749578, 0.851897946484132, 0.8502089074584408, 0.8503867010400925, 0.8483420748510979, 0.848253178060272, 0.8455862743354965, 0.8455862743354965, 0.8441639256822828, 0.8445195128455862, 0.8437194417281536, 0.8439861321006311, 0.8425637834474176, 0.8423859898657658, 0.8413192283758556, 0.8408747444217264, 0.8403413636767713, 0.8401635700951195, 0.8398079829318161, 0.8397190861409903, 0.8386523246510801, 0.8389190150235577, 0.8388301182327318, 0.8389190150235577, 0.8366076984620855, 0.8370521824162148, 0.8366965952529114, 0.8371410792070406, 0.8373188727886923, 0.8374077695795181, 0.8373188727886923, 0.8374077695795181, 0.8375855631611698, 0.837496666370344, 0.8375855631611698, 0.8376744599519957, 0.8373188727886923, 0.8377633567428215, 0.8372299759978665, 0.8379411503244732, 0.8368743888345631, 0.8368743888345631, 0.8360743177171304, 0.8362521112987821, 0.8350075562272202, 0.8353631433905236, 0.8342963819006134, 0.8353631433905236, 0.835096453018046, 0.8352742465996977, 0.8341185883189617, 0.8349186594363943, 0.8347408658547426, 0.8348297626455685, 0.8349186594363943, 0.8353631433905236, 0.8349186594363943, 0.8349186594363943, 0.8349186594363943, 0.835096453018046, 0.8352742465996977, 0.8355409369721754, 0.8355409369721754, 0.8351853498088719, 0.8351853498088719, 0.8353631433905236, 0.8352742465996977, 0.8353631433905236, 0.8349186594363943, 0.835096453018046, 0.8352742465996977, 0.8349186594363943, 0.8345630722730909, 0.8344741754822651, 0.8337630011556583, 0.8337630011556583, 0.832696239665748, 0.8328740332473997, 0.8326073428749222, 0.8323406525024447, 0.8322517557116188, 0.8320739621299671, 0.8318072717574896, 0.8318961685483154, 0.8324295492932705, 0.8326073428749222, 0.8322517557116188, 0.8327851364565739, 0.8311849942217086, 0.831540581385012, 0.8294959551960174, 0.8300293359409725, 0.8286958840785847, 0.8287847808694107, 0.8266512578895902, 0.8271846386345453, 0.8247844252822473, 0.8254955996088541, 0.8222953151391235, 0.8227397990932528, 0.8193617210418704, 0.8197173082051737]
In [105]:
# plot the results (Manhattan distance)
plt.plot(k_potentials, grid_mean_scores)
plt.xlabel('Potential K values for KNN')
plt.ylabel('Cross-Validated Accuracy')
Out[105]:
<matplotlib.text.Text at 0x120cb5a50>

In [106]:
# plot the results (using Euclidean distance)
plt.plot(k_potentials, grid_mean_scores)
plt.xlabel('Potential K values for KNN')
plt.ylabel('Cross-Validated Accuracy')
Out[106]:
<matplotlib.text.Text at 0x12143b7d0>

In [107]:
# examine the best model
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
0.962396657481
{'n_neighbors': 1}
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
metric_params=None, n_jobs=1, n_neighbors=1, p=2,
weights='uniform')
In [ ]:

In [108]:
# read in the data & create matrices
X = X_scaled[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", \
"time_spend_company", "Work_accident", "promotion_last_5years", "low", "medium",\
"technical", "support", "IT", "product_mng", "marketing", "RandD", "accounting",\
"hr"]]
y = X_scaled["left"]
In [176]:
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
Support Vector Machines
In [177]:
# Train, fit and predict with SVM

from sklearn import svm
Model=svm.SVC(kernel='linear')
Model.fit(X_train,y_train)
Y_pred=Model.predict(X_test)
In [178]:
# Cross validation score for SVM
scores = cross_val_score(Model, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.780694712068
In [179]:
# Evaluate SVM for accuracy

from sklearn import metrics
metrics.accuracy_score(y_test,Y_pred)
Out[179]:
0.78106666666666669
In [180]:
# Confusion matrix for SVM

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2696  166]
[ 655  233]]
In [181]:
# Classification Accuracy confirmation between manual calculation and formula for SVM

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.781066666667
0.781066666667
In [182]:
# Misclassification Accuracy confirmation between manual calculation and formula for SVM
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.218933333333
0.218933333333
In [183]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated SVM

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.262387387387
0.262387387387
In [184]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for SVM

print(TN / float(TN + FP))
0.941998602376
In [185]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for SVM

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.583959899749
0.583959899749
In [186]:
# Misclassification Accuracy confirmation between manual calculation and formula for SVM
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.218933333333
0.218933333333
In [187]:
# Roc_curve for Support Vector Machine Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [92]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [93]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Support Vector Machines: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

Random Forest
In [157]:
# Random Forests Classifier from Scikit Learn

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_jobs=2)
RF.fit(X_train,y_train)
Y_pred=RF.predict(X_test)
In [158]:
# Score for 10-fold Cross Validation for Random Forests Classifier

scores = cross_val_score(RF, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.986309602501
In [159]:
# Prediction Accuracy for Random Forests

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.9896
In [160]:
# Confusion matrix for Random Forests

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2857    5]
[  34  854]]
In [161]:
# Classification Accuracy confirmation between manual calculation and formula for Random Forest

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.9896
0.9896
In [162]:
# Misclassification Accuracy confirmation between manual calculation and formula for Random Forest
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0104
0.0104
In [163]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for Random Forest Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.961711711712
0.961711711712
In [164]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for Random Forest Classifier

print(TN / float(TN + FP))
0.998252969951
In [165]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for Random Forest Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.994179278231
0.994179278231
In [166]:
# Misclassification Accuracy confirmation between manual calculation and formula for Random Forest
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0104
0.0104
In [167]:
# Roc_curve for Random Forest Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [168]:
y_pred_prob_Random_Forest = RF.predict_proba(X_test)[:, 1]
In [172]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_Random_Forest, bins=5)
plt.xlim(0, 1)
plt.title('Random Forest Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[172]:
<matplotlib.text.Text at 0x12013a8d0>

In [170]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [171]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Random Forest: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

Bagging
In [109]:
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
In [110]:
# Bagging Classifier

seed = 7
cart = DecisionTreeClassifier()
num_trees = 100
Bagging = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
Bagging.fit(X_train,y_train)
Y_pred=Bagging.predict(X_test)
In [111]:
# Cross Validation for Bagging

scores = cross_val_score(Bagging, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.987910076856
In [112]:
# Prediction Accuracy for Bagging

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.989333333333
In [113]:
# Confusion matrix for Bagging

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2854    8]
[  32  856]]
In [114]:
# Classification Accuracy confirmation between manual calculation and formula for Bagging

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.989333333333
0.989333333333
In [115]:
# Misclassification Accuracy confirmation between manual calculation and formula for Bagging
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0106666666667
0.0106666666667
In [116]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for Bagging Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.963963963964
0.963963963964
In [117]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for Bagging Classifier

print(TN / float(TN + FP))
0.997204751922
In [118]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for Bagging Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.990740740741
0.990740740741
In [119]:
# Misclassification Accuracy confirmation between manual calculation and formula for Bagging
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0106666666667
0.0106666666667
In [120]:
# Roc_curve for Bagging


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [121]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
In [122]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Bagging: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [123]:
y_pred_prob_Bagging = Bagging.predict_proba(X_test)[:, 1]
In [124]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_Bagging, bins=5)
plt.xlim(0, 1)
plt.title('Bagging Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[124]:
<matplotlib.text.Text at 0x11b217610>

In [ ]:

AdaBoost
In [125]:
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
In [126]:
# AdaBoost Classifier

seed = 7
cart = AdaBoostClassifier()
num_trees = 100
AdaBoostClassifier = AdaBoostClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
AdaBoostClassifier.fit(X_train,y_train)
Y_pred=AdaBoostClassifier.predict(X_test)
In [127]:
# Cross Validation for AdaBoost

scores = cross_val_score(AdaBoostClassifier, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.958576016468
In [128]:
# Prediction Accuracy for AdaBoost

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.958133333333
In [129]:
# Confusion matrix for AdaBoost

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2793   69]
[  88  800]]
In [130]:
# Classification Accuracy confirmation between manual calculation and formula for Bagging

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.958133333333
0.958133333333
In [131]:
# Misclassification Accuracy confirmation between manual calculation and formula for AdaBoost
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0418666666667
0.0418666666667
In [132]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for AdaBoost Boosting Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.900900900901
0.900900900901
In [133]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for AdaBoost Classifier

print(TN / float(TN + FP))
0.975890985325
In [134]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for AdaBoost Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.920598388953
0.920598388953
In [135]:
# Misclassification Accuracy confirmation between manual calculation and formula for AdaBoost
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0418666666667
0.0418666666667
In [136]:
# Roc_curve for AdaBoost


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [137]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
In [138]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='AdaBoost: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [139]:
y_pred_prob_AdaBoost = AdaBoostClassifier.predict_proba(X_test)[:, 1]
In [140]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_AdaBoost, bins=5)
plt.xlim(0, 1)
plt.title('AdaBoost Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[140]:
<matplotlib.text.Text at 0x11b367a10>

In [ ]:

In [ ]:

In [ ]:

Gradient Boosting
In [141]:
from sklearn.ensemble import GradientBoostingClassifier
In [142]:
# Train and predict Gradient Boosting

GradientBoosting = GradientBoostingClassifier()
GradientBoosting.fit(X_train,y_train)
Y_pred=GradientBoosting.predict(X_test)
In [143]:
# Cross Validation for Gradient Boosting

scores = cross_val_score(GradientBoosting, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.976443246546
In [144]:
# Prediction Accuracy for Gradient Boosting

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.974133333333
In [145]:
# Confusion matrix for Gradient Boosting

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2832   30]
[  67  821]]
In [146]:
# Classification Accuracy confirmation between manual calculation and formula for Gradient Boosting

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.974133333333
0.974133333333
In [147]:
# Misclassification Accuracy confirmation between manual calculation and formula for Gradient Boosting
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0258666666667
0.0258666666667
In [148]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for Gradient Boosting Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.92454954955
0.92454954955
In [149]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for Gradient Classifier

print(TN / float(TN + FP))
0.989517819706
In [150]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for Gradient Boosting Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.964747356052
0.964747356052
In [151]:
# Misclassification Accuracy confirmation between manual calculation and formula for Gradient Boosting
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0258666666667
0.0258666666667
In [152]:
# Roc_curve for Gradient Boosting


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Gradient Boosting Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [153]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
In [154]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Gradient Boosting: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [155]:
y_pred_prob_GradientBoosting = GradientBoosting.predict_proba(X_test)[:, 1]
In [156]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_GradientBoosting, bins=5)
plt.xlim(0, 1)
plt.title('Gradient Boosting Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[156]:
<matplotlib.text.Text at 0x11c2566d0>

In [ ]:

In [ ]:

Part IV
Conclusion
One interesting chart showed us which departments have the most people leaving (and staying). They are Sales followed by Technical and Support. That means that the biggest effort at the company should be placed on keeping the strong employees in those departments (but by no means neglect the other departments).
There are several other interesting patterns in the data. Some are very intuitive whereas others very surprising and unexpected. There are three clusters of people leaving the company (the most important cluster is not as dense, in the sense that there are actually slightly more people there that stay than leave, but nevertheless its the most interesting and surprising cluster).
The three groups are employees with high performance and low satisfaction, employees with low satisfcation and performance (rather obvious) and employees with high satisfaction and performance (a surprising result). The first two results are very strong but rather trivial, there is limited use in providing detailed analyses of them. However, the somewhat less dense but very significant group of highly satisfied employees that do remarkably well in their evaluations is of extreme importance. Why do they leave? Well, there are several important factors to consider.
The people with high salaries and at even average satisfaction levels rarely leave the firm. Hence, one attribute that people who tend to leave will generally possess is salaries that are not high and fall in the low or medium range. It is interesting to note though that people with near perfect to perfect satisfaction levels do not leave even if they are in the low or medium income range brackets. Hence, a possible course of action to potentially alleviate this issue is increase salaries for good employees. Also, a further investigation into what might be the differentiating factor between people who are very satisfied (0.7-0.9 rate) vs super satisfied (0.9-1.0) can be done to see if it is possible to further increase the satisfaction level of those that are very satisfied but not perfectly so (maybe its possible to do this while at the same time saving money by not instituting as much pay increases if there is a way to get them satisfied in another manner, perhaps by working less hours for instance) would be a good idea.
Secondly, promotions are a huge factor. Looking at the plot of promotions during the past five years, it is very clear that people who are promoted stay and people who are not promoted leave. Of course, promotions and salary increases have some correlation because people tend to receive salary increases during promotions. Hence, it's important to promote employees if they do good work! That will help decrease the turnover rate.
The third important point is that employees who work a lot of hours that stay at the company for at least a medium amount of time tend to leave. Only the relatively small number of employees who work at the company for many years stay at the company regardless of working hours but very few make it there. Hence, the hours worked plays an important decision making role when employees think about whether to stay at the firm or seek other employment. Due to this finding it is important to see if any kind of rebalancing can be done so that part of the work can perhaps be shifted between employees who do not work as many hours and those that do in order to make the distribution of hours worked by employees more uniform which can perhaps increase employees willingness to stay at the company longer. It is quiet possible that some employees just feel burnt out even if they are generally happy with their job. Due to an inefficient work life balance, they might seek to find employment elsewhere.
Multiple machine learning algorithms were ran on the data and quiet a few did relatively well in terms of making predictions on out of sample data as measured by the accuracy score, the confusion matrix and deriatives thereof as well as the ROC Curve. In particular, the Random Forest and Bagging algorithms did the best with accuracy and cross-validation scores marginally different from each other just under 99%. Both correctly predicted over 3,700 out of 3,750 employees accurately (compared to a baseline accuracy of only 76% (2,850) as to whether they would leave/stay). The various scores computed from the confusion matrix were all almost perfect. Particularly importantly, the ROC curve has an area covering of .98 for both those algorithms.
Hence, using either of these two algorithms (Random Forests or Bagging) have given us great predictions as to which employees will leave vs which ones would stay.
Assumptions
We have assumed the dataset is accurate as we have not audited its validity.
Another assumption we made is that we trained on 75% of the data, conducted cross-validation, and then tested on the 25% remaining out of sample. However, when trying to sample to train a larger sample of the data (such as 90/10 test/train split), we received very similar results in our algorithms. Hence, our model is robust as should perform well when used on future data.
An assumption made by the Random Forest Model which did exceptionally well is that the sampling that it conducted during boostrap aggregation is actually representative. This is a safe assumption to make and the model has done really well on out of sample data.
Introduction:
This project analyzes a dataset in an attempt to determine what causes the good employees to leave their current jobs. The first step is to clean the data (Part I) and visualize it (Part II) using various types of graphs and charts to help establish clear relationships between various features. Part III analyzes the dataset using various Machine Learning algorithms and picks the best algorithm to model this dataset based on various metrics. Lastly, a conclusion on the entire analysis is provided.
Part I
Data Cleaning
This initial stage of the project is to clean the data in such a way that all the important attributes of the dataset be retained and unhindered for analysis purposes. Hence, the initial cleaning phase of the project imports the dataset and the libraries needed for the cleaning stages of the project. A check for Null Values/Missing Values is conducted (none are present). The multiple categorical variables in the dataset are created into dummy variables which is necessary for later stages of the analysis. These new variables are afterwards added to the original dataset and the initial variables from which dummification was done are removed to avoid redundancy. Lastly, the dataset is normalized due to the need to apply machine learning algorithms to it in later stages of the project.
In [1]:
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

%matplotlib inline
In [2]:
HR = pd.read_csv('HR_comma_sep.csv')
HR.head(10)
Out[2]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary
0	0.38	0.53	2	157	3	0	1	0	sales	low
1	0.80	0.86	5	262	6	0	1	0	sales	medium
2	0.11	0.88	7	272	4	0	1	0	sales	medium
3	0.72	0.87	5	223	5	0	1	0	sales	low
4	0.37	0.52	2	159	3	0	1	0	sales	low
5	0.41	0.50	2	153	3	0	1	0	sales	low
6	0.10	0.77	6	247	4	0	1	0	sales	low
7	0.92	0.85	5	259	5	0	1	0	sales	low
8	0.89	1.00	5	224	5	0	1	0	sales	low
9	0.42	0.53	2	142	3	0	1	0	sales	low
In [3]:
HR[HR['satisfaction_level'] == np.nan]
HR[HR['last_evaluation'] == np.nan]
HR[HR['number_project'] == np.nan]
HR[HR['average_montly_hours'] == np.nan]
HR[HR['time_spend_company'] == np.nan]
HR[HR['Work_accident'] == np.nan]
HR[HR['left'] == np.nan]
HR[HR['sales'] == np.nan]
HR[HR['salary'] == np.nan]
Out[3]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary
In [4]:
# Create a heatmap, to check for null values.

plt.figure(figsize=(10,7))
sns.heatmap(HR.isnull(),yticklabels=False, cbar=False, cmap='viridis')
Out[4]:
<matplotlib.axes._subplots.AxesSubplot at 0x114c6ee50>

In [5]:
HR['sales'].value_counts()
Out[5]:
sales          4140
technical      2720
support        2229
IT             1227
product_mng     902
marketing       858
RandD           787
accounting      767
hr              739
management      630
Name: sales, dtype: int64
In [6]:
HR['salary'].value_counts()
Out[6]:
low       7316
medium    6446
high      1237
Name: salary, dtype: int64
In [7]:
HR['left'].value_counts()
Out[7]:
0    11428
1     3571
Name: left, dtype: int64
In [8]:
# Create dummy variables for all the departments

sales = pd.get_dummies(HR.sales)

sales.columns = ["IT", "RandD", "accounting", "hr", "management", "marketing", "product_mng", "sales_department", "support", "technical"]

sales.head()
Out[8]:
IT	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical
0	0	0	0	0	0	0	0	1	0	0
1	0	0	0	0	0	0	0	1	0	0
2	0	0	0	0	0	0	0	1	0	0
3	0	0	0	0	0	0	0	1	0	0
4	0	0	0	0	0	0	0	1	0	0
In [9]:
# Create dummy variables for the salary

salary = pd.get_dummies(HR.salary)

salary.columns = ['low', 'medium', 'high']

salary.head()
Out[9]:
low	medium	high
0	0	1	0
1	0	0	1
2	0	0	1
3	0	1	0
4	0	1	0
In [10]:
# Create dummy variable for stayed/left

left = pd.get_dummies(HR.left)

left.columns = ['left', 'stayed']

left.head(10)
Out[10]:
left	stayed
0	0	1
1	0	1
2	0	1
3	0	1
4	0	1
5	0	1
6	0	1
7	0	1
8	0	1
9	0	1
In [11]:
# Concatenate the original dataset with the newly created salary and sales dummified variables

HR = pd.concat([HR, salary, sales], axis = 1)

HR.head(10)
Out[11]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary	...	IT	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical
0	0.38	0.53	2	157	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
1	0.80	0.86	5	262	6	0	1	0	sales	medium	...	0	0	0	0	0	0	0	1	0	0
2	0.11	0.88	7	272	4	0	1	0	sales	medium	...	0	0	0	0	0	0	0	1	0	0
3	0.72	0.87	5	223	5	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
4	0.37	0.52	2	159	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
5	0.41	0.50	2	153	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
6	0.10	0.77	6	247	4	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
7	0.92	0.85	5	259	5	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
8	0.89	1.00	5	224	5	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
9	0.42	0.53	2	142	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
10 rows × 23 columns
In [12]:
HR.dtypes
Out[12]:
satisfaction_level       float64
last_evaluation          float64
number_project             int64
average_montly_hours       int64
time_spend_company         int64
Work_accident              int64
left                       int64
promotion_last_5years      int64
sales                     object
salary                    object
low                        uint8
medium                     uint8
high                       uint8
IT                         uint8
RandD                      uint8
accounting                 uint8
hr                         uint8
management                 uint8
marketing                  uint8
product_mng                uint8
sales_department           uint8
support                    uint8
technical                  uint8
dtype: object
In [13]:
# Create a new column 

salary_map = {'low': 1, 'medium': 2, 'high': 3}

HR['salary_variable'] = HR['salary'].apply(lambda x: salary_map[x])

HR.head()
Out[13]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary	...	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical	salary_variable
0	0.38	0.53	2	157	3	0	1	0	sales	low	...	0	0	0	0	0	0	1	0	0	1
1	0.80	0.86	5	262	6	0	1	0	sales	medium	...	0	0	0	0	0	0	1	0	0	2
2	0.11	0.88	7	272	4	0	1	0	sales	medium	...	0	0	0	0	0	0	1	0	0	2
3	0.72	0.87	5	223	5	0	1	0	sales	low	...	0	0	0	0	0	0	1	0	0	1
4	0.37	0.52	2	159	3	0	1	0	sales	low	...	0	0	0	0	0	0	1	0	0	1
5 rows × 24 columns
In [14]:
del HR['salary']

del HR['sales']
In [15]:
HR.head()
Out[15]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	low	medium	...	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical	salary_variable
0	0.38	0.53	2	157	3	0	1	0	0	1	...	0	0	0	0	0	0	1	0	0	1
1	0.80	0.86	5	262	6	0	1	0	0	0	...	0	0	0	0	0	0	1	0	0	2
2	0.11	0.88	7	272	4	0	1	0	0	0	...	0	0	0	0	0	0	1	0	0	2
3	0.72	0.87	5	223	5	0	1	0	0	1	...	0	0	0	0	0	0	1	0	0	1
4	0.37	0.52	2	159	3	0	1	0	0	1	...	0	0	0	0	0	0	1	0	0	1
5 rows × 22 columns
In [16]:
HR.dtypes
Out[16]:
satisfaction_level       float64
last_evaluation          float64
number_project             int64
average_montly_hours       int64
time_spend_company         int64
Work_accident              int64
left                       int64
promotion_last_5years      int64
low                        uint8
medium                     uint8
high                       uint8
IT                         uint8
RandD                      uint8
accounting                 uint8
hr                         uint8
management                 uint8
marketing                  uint8
product_mng                uint8
sales_department           uint8
support                    uint8
technical                  uint8
salary_variable            int64
dtype: object
In [17]:
from sklearn.preprocessing import MinMaxScaler
X_scaled = MinMaxScaler().fit_transform(HR.astype(float))
X_scaled = pd.DataFrame(X_scaled, columns = HR.columns)

# Min Max Scaler

X_scaled.head(10)
Out[17]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	low	medium	...	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical	salary_variable
0	0.318681	0.265625	0.0	0.285047	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
1	0.780220	0.781250	0.6	0.775701	0.500	0.0	1.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.5
2	0.021978	0.812500	1.0	0.822430	0.250	0.0	1.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.5
3	0.692308	0.796875	0.6	0.593458	0.375	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
4	0.307692	0.250000	0.0	0.294393	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
5	0.351648	0.218750	0.0	0.266355	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
6	0.010989	0.640625	0.8	0.705607	0.250	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
7	0.912088	0.765625	0.6	0.761682	0.375	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
8	0.879121	1.000000	0.6	0.598131	0.375	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
9	0.362637	0.265625	0.0	0.214953	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
10 rows × 22 columns
Part II
Exploratory Data Analysis (EDA)
In this section, we try to visualize the data on employees using various plots, graphs, charts in order to get an idea of how different features in the dataset relate to each other.
The below represents a heatmap of correlations of features. Some interesting observations from this heatmap: Quiet intuitively, you can see that the relationship between satisfaction level and people that left the company is negative. It's also interesting to note that there is a positive correlation between the number of hours employees work and their evaluation. These and other relationships will be explored more deeply in the rest of the EDA.
In [18]:
correlation = X_scaled.corr()
plt.figure(figsize=(20,20))
sns.heatmap(correlation, vmax=10, square=True,annot=True,cmap=None, linewidths=1)

plt.title('Correlation between features')
Out[18]:
<matplotlib.text.Text at 0x118f3a050>

In [19]:
# Exported csv file to work with in Tableau
X_scaled.to_csv("Updated_Capstone.csv")
In [20]:
X_scaled['left'].value_counts()
Out[20]:
0.0    11428
1.0     3571
Name: left, dtype: int64
In [21]:
stayed = np.where(X_scaled['left'] == 0)[0]
left = np.where(X_scaled['left'] == 1)[0]
For a given level of satisfaction, there is definitely an intuitive pattern that the higher someone's salary is, the less they are likely to leave. This relationship is a lot more apparent in the distinction between the high income earners vs low & median income earners as a group. The difference in the people that left between low and median income salaries is not significant. Hence, most of the people that leave are not receiving a high salary, which means that you if you would like to retain these employees, you should increase their pay!
In [22]:
# plot of Satisfaction vs Last Evaluation
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.satisfaction_level[stayed], X_scaled.salary_variable[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.satisfaction_level[left], X_scaled.salary_variable[left], alpha=0.1, color='r')
ax.set_title("satisfaction_level vs salary")
ax.set_xlabel("satisfaction_level")
ax.set_ylabel("salary")
plt.show()

In the Satisfaction vs Last Evaluation plot below, we can note some interesting observations. Firstly, there is a sizeable cluster of people with strong evaluations and low satisfaction levels leaving. Secondly, there is a significant group of people leaving that are slighly below average in performance and evaluation. Lastly, and by far the most interesting observation of the three, there is a decent size of a not very dense cluster of people leaving that have done extremely well on their evaluations (0.7 to 1 on a 0-1 scale) and and have very high satisfaction rates, albeit not perfect (0.7-0.9 on a 0-1 scale). That particular group of employees also have many people that stay, but a significant proportion of that group is leaving. So the most surprising thing to observe is that people can be satisfied with the job and still leave! There are several possibilites for why this might be possible. People might enjoy their job but think they can do even better elsewhere, whether its income or career growth (or a combination of the two, considering that those things tend to go hand in hand). Another possibility is that people might not be honest in surveys because of fear of being reprimanded for negative surveys.
In [23]:
# plot of Satisfaction vs Last Evaluation
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.satisfaction_level[stayed], X_scaled.last_evaluation[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.satisfaction_level[left], X_scaled.last_evaluation[left], alpha=0.1, color='r')
ax.set_title("satisfaction_level vs last_evaluation")
ax.set_xlabel("satisfaction_level")
ax.set_ylabel("last_evaluation")
plt.show()

The Satisfaction vs Promotion during the last five years plot below shows a very strong relationship in several interesting ways! People who have been promoted don't leave very often! However, such promotions are not common enough! Therefore, if you don't want good employees to leave, promoting them will really help!
In [24]:
# plot of Satisfaction vs Promotion during last five years
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.satisfaction_level[stayed], X_scaled.promotion_last_5years[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.satisfaction_level[left], X_scaled.promotion_last_5years[left], alpha=0.1, color='r')
ax.set_title("satisfaction_level vs promotion_last_5years")
ax.set_xlabel("satisfaction_level")
ax.set_ylabel("promotion_last_5years")
plt.show()

The below scatter plot compares the average working hours against the time spent at the company and how it impacts the employee turnover rate. There are multiple interesting relationships to note. For employees that just started out, the majority tend to stick around for some time even if they work substantial hours. However, for employees who have been at the company for a while, there is a significant tendency to leave the company if working substantial hours. Consequently, we can observe that very few employees stick around at the company for a long time period because the ones who were working long hours almost all left! However, those that do stick around for a long time (relatively few) tend to all be devoted and not leave
There are several conjectures for why this is possible. One is that employees have worked for a significant amount of hours and put in a lot of time and effort expecting a quick promotion and/or pay increase (refer to graph above for relationship between promotions and staying/leaving) that they have not received and leave for elsewhere. Another reason might be that people have tried working for a little bit of time, realized that they are being overworked and have started looking for other opportunities to seek employment.
In [25]:
# plot of Average_monthly_hours vs time_spend_company
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.average_montly_hours[stayed], X_scaled.time_spend_company[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.average_montly_hours[left], X_scaled.time_spend_company[left], alpha=0.1, color='r')
ax.set_title("average_monthly_hours vs time_spend_company")
ax.set_xlabel("average_monthly_hours")
ax.set_ylabel("time_spend_company")
plt.show()

In [26]:
# Imported from Tableau
from IPython.display import Image
Image(filename = 'Average Monthly Hours vs Satisfaction Level.png', width = 1000)
Out[26]:

There does not appear to be a strong relationship between accidents and people leaving the company, hence, people stay/leave regardless of the accidents they might have experienced.
In [27]:
# plot of work_accident vs time_spend_company
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.Work_accident[stayed], X_scaled.time_spend_company[stayed], alpha=0.5, color='g')
ax.scatter(X_scaled.Work_accident[left], X_scaled.time_spend_company[left], alpha=0.5, color='r')
ax.set_title("Work_accident vs time_spend_company")
ax.set_xlabel("Work_accident")
ax.set_ylabel("time_spend_company")
plt.show()

The below stripplot compares number of projects done by an employee versus the number of promotions they have received over the past 5 years. It is interesting to note that the number of projects completed has very insignificant impact on whether a person leaves. On the contrary, the promotions received during the past 5 years is of critical importance. The majority of people who have received a promotion stay. Almost all those who were not promoted leave.
The data seems insufficient to be able to determine why the number of projects is not related to promotions. There are at least several possibilities. Perhaps the number of projects refers to the number assigned (vs completed) or that the projects are not all of equal difficulty and not equally time consuming leading to distorted hypothesis that the number of projects might necessarily correspond to an employee's productivity.
In [28]:
import seaborn as sns

sns.stripplot("number_project", "promotion_last_5years", data=X_scaled, hue="left", jitter = True)
Out[28]:
<matplotlib.axes._subplots.AxesSubplot at 0x117949d90>

The Satisfaction level distribution below shows that the majority of people are at least moderately satisfied with their job. Therefore, for the people that are satisfied and do well, efforts should be placed on making them even more happy by promoting them. For the people that do poorly or mediocre, either studies should be placed on determining why they are not doing well and trying to improve their performance or attempting to replace them with better employees (this would depend on a cost/benefit analysis) done by the company regarding this issue.
In [29]:
# Histogram distribution of Satisfaction Level

f,ax1 = plt.subplots(1,1)
sns.distplot(X_scaled['satisfaction_level'],bins=10,norm_hist=False)
plt.show()

The distribution (which is a decent approximation of a normal distribution with almost no tails and some kurtosis present) of the last evaluation graph shows us there is a sizeable portion of people that do well in their evaluations, hence the company should try to do as much as possible to keep those people!
In [30]:
# Histogram distribution of Last Evaluation


f,ax1 = plt.subplots(1,1)
sns.distplot(X_scaled['last_evaluation'],bins=10,norm_hist=False)
plt.show()

The below plot shows the ratios of people who stay in the company by department and the ratio of people who leave the company by department. It is clear from the chart that the departments with the most number of employees also have the highest ratios of both people who have left and those that stayed at the company. This tells us there should be particular effort place on retaining good employees that work in those departments, in particular the sales, technical and support departments, respectively. The largest efforts should be placed on retaining the good employees in these groups because they have the most people leaving.
In [34]:
# This plot is to compare the people that left across different departments
# The non-department variables are deleted to only keep the departments

del X_scaled['satisfaction_level']
del X_scaled['last_evaluation']
del X_scaled['number_project']
del X_scaled['average_montly_hours']
del X_scaled['time_spend_company']
del X_scaled['Work_accident']
del X_scaled['promotion_last_5years']
del X_scaled['low']
del X_scaled['medium']
del X_scaled['high']
del X_scaled['salary_variable']


left=X_scaled.groupby('left').mean()
left=left.transpose()
plt.figure(figsize=(20,10))
left.plot(kind='bar')
Out[34]:
<matplotlib.axes._subplots.AxesSubplot at 0x11a7bcd10>
<matplotlib.figure.Figure at 0x11a360850>

Part III
Predictive Modelling
In this section below, multiple models will be run that are applicable to classification. Afterwards, various scoring metrics will be calculated for the models and a determination will be made as to which model is based for this dataset.
In [174]:
# read in the data & create matrices
X = X_scaled[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", \
"time_spend_company", "Work_accident", "promotion_last_5years", "salary_variable",\
"technical", "support", "IT", "product_mng", "marketing", "RandD", "accounting",\
"hr", "management", "sales_department"]]
y = X_scaled["left"]
In [175]:
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
Logistic Regression Model
In [33]:
# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
model = LogisticRegression()

# fit the model with data
mdl = model.fit(X_train, y_train)

# predict the response for new observations

logit = model.predict(X_test)

# sklearn output to check that ran on right data split.

len(logit)
Out[33]:
3750
In [34]:
# These represent the y-intercept and coefficients for all the variables in the logistic regression.

print(model.intercept_)
print(model.coef_)
[ 0.57305398]
[[-3.71487625  0.45484967 -1.50619896  0.8704181   1.97590531 -1.45425383
-1.26024249 -1.32512177  0.22588571  0.23497924 -0.01282827  0.07509077
0.13925062 -0.47542127  0.26606552  0.3920348  -0.44451555  0.17251241]]
In [35]:
# Prediction Accuracy for Logistic Regression

from sklearn import metrics
print(metrics.accuracy_score(y_test, logit))
0.796
In [36]:
from sklearn.cross_validation import cross_val_score
In [37]:
# 10-fold cross-validation for Logistic Regression to find the CV score
mdl = model.fit(X_train, y_train)
scores = cross_val_score(mdl, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.794205514996
In [38]:
from sklearn import metrics
confusion = metrics.confusion_matrix(y_test, logit)
print ((confusion))
[[2651  211]
[ 554  334]]
In [39]:
y_pred_prob = model.predict_proba(X_test)[:, 1]
In [40]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob, bins=5)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[40]:
<matplotlib.text.Text at 0x1181a6a90>

In [41]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [42]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, logit)


plot_confusion_matrix(cm, title='LogisticRegression: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [43]:
# Roc_curve for Logistic Regression Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, logit)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Logistic Regression Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

K Nearest Nearbours Model
K=1
In [44]:
from sklearn.neighbors import KNeighborsClassifier
In [45]:
# K-Nearest Neighbours with K=1
knn = KNeighborsClassifier(n_neighbors=1, metric = 'euclidean')
In [46]:
y = knn.fit(X_train, y_train)

KNN = knn.predict(X_test)

# sklearn output to check that ran on right data split.

len(KNN)
Out[46]:
3750
In [47]:
# Cross fold validation for K=1

md2 = knn.fit(X_train, y_train)
scores = cross_val_score(md2, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.962043001151
In [48]:
# Prediction Accuracy.

from sklearn import metrics
print(metrics.accuracy_score(y_test, KNN))
0.967466666667
In [49]:
confusion = metrics.confusion_matrix(y_test, KNN)
print confusion
[[2775   87]
[  35  853]]
In [50]:
# Classification Accuracy confirmation between manual calculation and formula for K=1

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, KNN))
0.967466666667
0.967466666667
In [51]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=1
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN))
0.0325333333333
0.0325333333333
In [52]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for K=1

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, KNN))
0.960585585586
0.960585585586
In [53]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for K=1


print(TN / float(TN + FP))
0.969601677149
In [54]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for K=1

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, KNN))
0.907446808511
0.907446808511
In [55]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=1
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN))
0.0325333333333
0.0325333333333
In [56]:
y_pred_prob_knn = knn.predict_proba(X_test)[:, 1]
In [57]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_knn, bins=5)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[57]:
<matplotlib.text.Text at 0x1181d6b90>

In [58]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [59]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, KNN)


plot_confusion_matrix(cm, title='KNN for K=1: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [60]:
# Roc_curve for KNN=1 Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, KNN)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('KNN=1 Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

K=5
In [61]:
# K-Nearest Neighbours with K=5

knn = KNeighborsClassifier(n_neighbors=5, metric = 'euclidean')
y = knn.fit(X_train, y_train)
KNN5 = knn.predict(X_test)

# sklearn output to check that ran on right data split.

len(KNN5)
Out[61]:
3750
In [62]:
# Cross Validation for K=5 Cross-Fold Validation

md2 = knn.fit(X_train, y_train)
scores = cross_val_score(md2, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.938309179837
In [63]:
# Prediction Accuracy on KNN=5 for accuracy score

from sklearn import metrics
print(metrics.accuracy_score(y_test, KNN5))
0.941333333333
In [64]:
confusion = metrics.confusion_matrix(y_test, KNN5)
print confusion
[[2731  131]
[  89  799]]
In [65]:
# Classification Accuracy confirmation between manual calculation and formula for K=5

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, KNN5))
0.941333333333
0.941333333333
In [66]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=5
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN5))
0.0586666666667
0.0586666666667
In [67]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for K=5

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, KNN5))
0.899774774775
0.899774774775
In [68]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for K=5

print(TN / float(TN + FP))
0.954227812718
In [69]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for K=5

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, KNN5))
0.859139784946
0.859139784946
In [70]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=5
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN5))
0.0586666666667
0.0586666666667
In [71]:
# Roc_curve for KNN=5 Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, KNN5)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [72]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [73]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, KNN5)


plot_confusion_matrix(cm, title='KNN for N=5: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [74]:
y_pred_prob_knn = knn.predict_proba(X_test)[:, 1]
In [75]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_knn, bins=5)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[75]:
<matplotlib.text.Text at 0x11acdea10>

In [ ]:

Grid Search for optimal N value in KNN
In [101]:
from sklearn.grid_search import GridSearchCV
/Users/Misha/anaconda/lib/python2.7/site-packages/sklearn/grid_search.py:43: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.
DeprecationWarning)
In [102]:
k_potentials = list(range(1, 200))
print(k_potentials)
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]
In [103]:
# Set Nearest Neighbours equal to k
knn = KNeighborsClassifier(n_neighbors=k_potentials, metric = 'euclidean')
# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_potentials)
# instantiate the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
# fit the grid with data
grid.fit(X_train, y_train)
# view the results
grid.grid_scores_
Out[103]:
[mean: 0.96240, std: 0.00591, params: {'n_neighbors': 1},
mean: 0.95884, std: 0.00543, params: {'n_neighbors': 2},
mean: 0.94568, std: 0.00463, params: {'n_neighbors': 3},
mean: 0.94622, std: 0.00555, params: {'n_neighbors': 4},
mean: 0.93875, std: 0.00620, params: {'n_neighbors': 5},
mean: 0.94284, std: 0.00673, params: {'n_neighbors': 6},
mean: 0.93795, std: 0.00746, params: {'n_neighbors': 7},
mean: 0.93839, std: 0.00819, params: {'n_neighbors': 8},
mean: 0.93351, std: 0.00932, params: {'n_neighbors': 9},
mean: 0.93493, std: 0.00848, params: {'n_neighbors': 10},
mean: 0.93048, std: 0.00825, params: {'n_neighbors': 11},
mean: 0.93253, std: 0.00810, params: {'n_neighbors': 12},
mean: 0.92933, std: 0.00898, params: {'n_neighbors': 13},
mean: 0.92968, std: 0.00824, params: {'n_neighbors': 14},
mean: 0.92719, std: 0.00747, params: {'n_neighbors': 15},
mean: 0.92817, std: 0.00830, params: {'n_neighbors': 16},
mean: 0.92559, std: 0.00873, params: {'n_neighbors': 17},
mean: 0.92728, std: 0.00821, params: {'n_neighbors': 18},
mean: 0.92497, std: 0.00821, params: {'n_neighbors': 19},
mean: 0.92470, std: 0.00830, params: {'n_neighbors': 20},
mean: 0.92204, std: 0.00804, params: {'n_neighbors': 21},
mean: 0.92186, std: 0.00867, params: {'n_neighbors': 22},
mean: 0.91902, std: 0.00856, params: {'n_neighbors': 23},
mean: 0.91777, std: 0.00850, params: {'n_neighbors': 24},
mean: 0.91564, std: 0.00809, params: {'n_neighbors': 25},
mean: 0.91457, std: 0.00913, params: {'n_neighbors': 26},
mean: 0.91244, std: 0.00927, params: {'n_neighbors': 27},
mean: 0.91244, std: 0.00873, params: {'n_neighbors': 28},
mean: 0.91101, std: 0.00878, params: {'n_neighbors': 29},
mean: 0.91110, std: 0.00853, params: {'n_neighbors': 30},
mean: 0.90906, std: 0.00855, params: {'n_neighbors': 31},
mean: 0.90941, std: 0.00906, params: {'n_neighbors': 32},
mean: 0.90773, std: 0.00922, params: {'n_neighbors': 33},
mean: 0.90728, std: 0.00991, params: {'n_neighbors': 34},
mean: 0.90550, std: 0.00946, params: {'n_neighbors': 35},
mean: 0.90417, std: 0.00891, params: {'n_neighbors': 36},
mean: 0.90292, std: 0.00887, params: {'n_neighbors': 37},
mean: 0.90097, std: 0.00881, params: {'n_neighbors': 38},
mean: 0.90035, std: 0.00948, params: {'n_neighbors': 39},
mean: 0.89875, std: 0.01009, params: {'n_neighbors': 40},
mean: 0.89759, std: 0.01009, params: {'n_neighbors': 41},
mean: 0.89626, std: 0.00955, params: {'n_neighbors': 42},
mean: 0.89537, std: 0.01016, params: {'n_neighbors': 43},
mean: 0.89270, std: 0.01221, params: {'n_neighbors': 44},
mean: 0.89172, std: 0.01244, params: {'n_neighbors': 45},
mean: 0.88950, std: 0.01089, params: {'n_neighbors': 46},
mean: 0.88906, std: 0.01150, params: {'n_neighbors': 47},
mean: 0.88666, std: 0.01137, params: {'n_neighbors': 48},
mean: 0.88577, std: 0.01143, params: {'n_neighbors': 49},
mean: 0.88523, std: 0.01132, params: {'n_neighbors': 50},
mean: 0.88452, std: 0.01149, params: {'n_neighbors': 51},
mean: 0.88372, std: 0.01237, params: {'n_neighbors': 52},
mean: 0.88283, std: 0.01227, params: {'n_neighbors': 53},
mean: 0.88141, std: 0.01166, params: {'n_neighbors': 54},
mean: 0.88114, std: 0.01085, params: {'n_neighbors': 55},
mean: 0.88061, std: 0.01041, params: {'n_neighbors': 56},
mean: 0.87990, std: 0.01105, params: {'n_neighbors': 57},
mean: 0.87910, std: 0.00974, params: {'n_neighbors': 58},
mean: 0.87892, std: 0.00997, params: {'n_neighbors': 59},
mean: 0.87803, std: 0.01108, params: {'n_neighbors': 60},
mean: 0.87768, std: 0.01119, params: {'n_neighbors': 61},
mean: 0.87688, std: 0.01152, params: {'n_neighbors': 62},
mean: 0.87652, std: 0.01182, params: {'n_neighbors': 63},
mean: 0.87590, std: 0.01129, params: {'n_neighbors': 64},
mean: 0.87554, std: 0.01148, params: {'n_neighbors': 65},
mean: 0.87474, std: 0.01183, params: {'n_neighbors': 66},
mean: 0.87412, std: 0.01223, params: {'n_neighbors': 67},
mean: 0.87394, std: 0.01196, params: {'n_neighbors': 68},
mean: 0.87341, std: 0.01210, params: {'n_neighbors': 69},
mean: 0.87314, std: 0.01231, params: {'n_neighbors': 70},
mean: 0.87270, std: 0.01219, params: {'n_neighbors': 71},
mean: 0.87226, std: 0.01168, params: {'n_neighbors': 72},
mean: 0.87154, std: 0.01185, params: {'n_neighbors': 73},
mean: 0.87074, std: 0.01104, params: {'n_neighbors': 74},
mean: 0.87057, std: 0.01132, params: {'n_neighbors': 75},
mean: 0.87003, std: 0.01129, params: {'n_neighbors': 76},
mean: 0.86968, std: 0.01148, params: {'n_neighbors': 77},
mean: 0.86754, std: 0.01056, params: {'n_neighbors': 78},
mean: 0.86692, std: 0.01041, params: {'n_neighbors': 79},
mean: 0.86568, std: 0.01028, params: {'n_neighbors': 80},
mean: 0.86594, std: 0.01024, params: {'n_neighbors': 81},
mean: 0.86461, std: 0.00937, params: {'n_neighbors': 82},
mean: 0.86390, std: 0.00915, params: {'n_neighbors': 83},
mean: 0.86239, std: 0.00956, params: {'n_neighbors': 84},
mean: 0.86257, std: 0.01001, params: {'n_neighbors': 85},
mean: 0.86088, std: 0.01023, params: {'n_neighbors': 86},
mean: 0.86088, std: 0.01051, params: {'n_neighbors': 87},
mean: 0.85981, std: 0.01214, params: {'n_neighbors': 88},
mean: 0.85937, std: 0.01208, params: {'n_neighbors': 89},
mean: 0.85830, std: 0.01087, params: {'n_neighbors': 90},
mean: 0.85821, std: 0.01108, params: {'n_neighbors': 91},
mean: 0.85616, std: 0.00995, params: {'n_neighbors': 92},
mean: 0.85652, std: 0.00954, params: {'n_neighbors': 93},
mean: 0.85519, std: 0.00892, params: {'n_neighbors': 94},
mean: 0.85545, std: 0.00957, params: {'n_neighbors': 95},
mean: 0.85368, std: 0.00900, params: {'n_neighbors': 96},
mean: 0.85332, std: 0.00845, params: {'n_neighbors': 97},
mean: 0.85199, std: 0.00923, params: {'n_neighbors': 98},
mean: 0.85190, std: 0.00926, params: {'n_neighbors': 99},
mean: 0.85021, std: 0.00825, params: {'n_neighbors': 100},
mean: 0.85039, std: 0.00930, params: {'n_neighbors': 101},
mean: 0.84834, std: 0.01015, params: {'n_neighbors': 102},
mean: 0.84825, std: 0.01016, params: {'n_neighbors': 103},
mean: 0.84559, std: 0.00954, params: {'n_neighbors': 104},
mean: 0.84559, std: 0.00922, params: {'n_neighbors': 105},
mean: 0.84416, std: 0.00977, params: {'n_neighbors': 106},
mean: 0.84452, std: 0.00978, params: {'n_neighbors': 107},
mean: 0.84372, std: 0.00935, params: {'n_neighbors': 108},
mean: 0.84399, std: 0.00922, params: {'n_neighbors': 109},
mean: 0.84256, std: 0.00943, params: {'n_neighbors': 110},
mean: 0.84239, std: 0.00930, params: {'n_neighbors': 111},
mean: 0.84132, std: 0.00941, params: {'n_neighbors': 112},
mean: 0.84087, std: 0.00924, params: {'n_neighbors': 113},
mean: 0.84034, std: 0.00864, params: {'n_neighbors': 114},
mean: 0.84016, std: 0.00875, params: {'n_neighbors': 115},
mean: 0.83981, std: 0.00849, params: {'n_neighbors': 116},
mean: 0.83972, std: 0.00862, params: {'n_neighbors': 117},
mean: 0.83865, std: 0.00779, params: {'n_neighbors': 118},
mean: 0.83892, std: 0.00782, params: {'n_neighbors': 119},
mean: 0.83883, std: 0.00798, params: {'n_neighbors': 120},
mean: 0.83892, std: 0.00810, params: {'n_neighbors': 121},
mean: 0.83661, std: 0.00577, params: {'n_neighbors': 122},
mean: 0.83705, std: 0.00582, params: {'n_neighbors': 123},
mean: 0.83670, std: 0.00582, params: {'n_neighbors': 124},
mean: 0.83714, std: 0.00545, params: {'n_neighbors': 125},
mean: 0.83732, std: 0.00565, params: {'n_neighbors': 126},
mean: 0.83741, std: 0.00610, params: {'n_neighbors': 127},
mean: 0.83732, std: 0.00647, params: {'n_neighbors': 128},
mean: 0.83741, std: 0.00634, params: {'n_neighbors': 129},
mean: 0.83759, std: 0.00626, params: {'n_neighbors': 130},
mean: 0.83750, std: 0.00609, params: {'n_neighbors': 131},
mean: 0.83759, std: 0.00639, params: {'n_neighbors': 132},
mean: 0.83767, std: 0.00616, params: {'n_neighbors': 133},
mean: 0.83732, std: 0.00640, params: {'n_neighbors': 134},
mean: 0.83776, std: 0.00641, params: {'n_neighbors': 135},
mean: 0.83723, std: 0.00690, params: {'n_neighbors': 136},
mean: 0.83794, std: 0.00675, params: {'n_neighbors': 137},
mean: 0.83687, std: 0.00710, params: {'n_neighbors': 138},
mean: 0.83687, std: 0.00718, params: {'n_neighbors': 139},
mean: 0.83607, std: 0.00744, params: {'n_neighbors': 140},
mean: 0.83625, std: 0.00725, params: {'n_neighbors': 141},
mean: 0.83501, std: 0.00739, params: {'n_neighbors': 142},
mean: 0.83536, std: 0.00761, params: {'n_neighbors': 143},
mean: 0.83430, std: 0.00774, params: {'n_neighbors': 144},
mean: 0.83536, std: 0.00776, params: {'n_neighbors': 145},
mean: 0.83510, std: 0.00746, params: {'n_neighbors': 146},
mean: 0.83527, std: 0.00732, params: {'n_neighbors': 147},
mean: 0.83412, std: 0.00716, params: {'n_neighbors': 148},
mean: 0.83492, std: 0.00704, params: {'n_neighbors': 149},
mean: 0.83474, std: 0.00740, params: {'n_neighbors': 150},
mean: 0.83483, std: 0.00759, params: {'n_neighbors': 151},
mean: 0.83492, std: 0.00751, params: {'n_neighbors': 152},
mean: 0.83536, std: 0.00788, params: {'n_neighbors': 153},
mean: 0.83492, std: 0.00833, params: {'n_neighbors': 154},
mean: 0.83492, std: 0.00841, params: {'n_neighbors': 155},
mean: 0.83492, std: 0.00815, params: {'n_neighbors': 156},
mean: 0.83510, std: 0.00822, params: {'n_neighbors': 157},
mean: 0.83527, std: 0.00826, params: {'n_neighbors': 158},
mean: 0.83554, std: 0.00832, params: {'n_neighbors': 159},
mean: 0.83554, std: 0.00778, params: {'n_neighbors': 160},
mean: 0.83519, std: 0.00745, params: {'n_neighbors': 161},
mean: 0.83519, std: 0.00757, params: {'n_neighbors': 162},
mean: 0.83536, std: 0.00715, params: {'n_neighbors': 163},
mean: 0.83527, std: 0.00767, params: {'n_neighbors': 164},
mean: 0.83536, std: 0.00778, params: {'n_neighbors': 165},
mean: 0.83492, std: 0.00759, params: {'n_neighbors': 166},
mean: 0.83510, std: 0.00790, params: {'n_neighbors': 167},
mean: 0.83527, std: 0.00820, params: {'n_neighbors': 168},
mean: 0.83492, std: 0.00775, params: {'n_neighbors': 169},
mean: 0.83456, std: 0.00746, params: {'n_neighbors': 170},
mean: 0.83447, std: 0.00782, params: {'n_neighbors': 171},
mean: 0.83376, std: 0.00770, params: {'n_neighbors': 172},
mean: 0.83376, std: 0.00731, params: {'n_neighbors': 173},
mean: 0.83270, std: 0.00737, params: {'n_neighbors': 174},
mean: 0.83287, std: 0.00736, params: {'n_neighbors': 175},
mean: 0.83261, std: 0.00701, params: {'n_neighbors': 176},
mean: 0.83234, std: 0.00676, params: {'n_neighbors': 177},
mean: 0.83225, std: 0.00719, params: {'n_neighbors': 178},
mean: 0.83207, std: 0.00718, params: {'n_neighbors': 179},
mean: 0.83181, std: 0.00746, params: {'n_neighbors': 180},
mean: 0.83190, std: 0.00741, params: {'n_neighbors': 181},
mean: 0.83243, std: 0.00808, params: {'n_neighbors': 182},
mean: 0.83261, std: 0.00827, params: {'n_neighbors': 183},
mean: 0.83225, std: 0.00819, params: {'n_neighbors': 184},
mean: 0.83279, std: 0.00779, params: {'n_neighbors': 185},
mean: 0.83118, std: 0.00688, params: {'n_neighbors': 186},
mean: 0.83154, std: 0.00684, params: {'n_neighbors': 187},
mean: 0.82950, std: 0.00716, params: {'n_neighbors': 188},
mean: 0.83003, std: 0.00689, params: {'n_neighbors': 189},
mean: 0.82870, std: 0.00649, params: {'n_neighbors': 190},
mean: 0.82878, std: 0.00704, params: {'n_neighbors': 191},
mean: 0.82665, std: 0.00624, params: {'n_neighbors': 192},
mean: 0.82718, std: 0.00654, params: {'n_neighbors': 193},
mean: 0.82478, std: 0.00895, params: {'n_neighbors': 194},
mean: 0.82550, std: 0.00846, params: {'n_neighbors': 195},
mean: 0.82230, std: 0.00974, params: {'n_neighbors': 196},
mean: 0.82274, std: 0.01041, params: {'n_neighbors': 197},
mean: 0.81936, std: 0.00926, params: {'n_neighbors': 198},
mean: 0.81972, std: 0.00963, params: {'n_neighbors': 199}]
In [104]:
# create a list of the mean scores only
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print(grid_mean_scores)
[0.9623966574806649, 0.9588407858476309, 0.9456840608054049, 0.94621744155036, 0.9387501111209885, 0.9428393634989777, 0.9379500400035559, 0.9383945239576851, 0.9335052004622633, 0.9349275491154769, 0.9304827095741843, 0.9325273357631789, 0.9293270512934483, 0.9296826384567517, 0.9271935283136279, 0.9281713930127122, 0.9255933860787625, 0.9272824251044537, 0.9249711085429816, 0.9247044181705041, 0.9220375144457285, 0.9218597208640769, 0.9190150235576495, 0.9177704684860877, 0.9156369455062672, 0.914570184016357, 0.9124366610365365, 0.9124366610365365, 0.911014312383323, 0.9111032091741488, 0.9090585829851542, 0.9094141701484576, 0.9077251311227664, 0.9072806471686372, 0.9055027113521202, 0.9041692594897324, 0.9029247044181705, 0.9009689750200017, 0.9003466974842208, 0.8987465552493555, 0.8975908969686194, 0.8962574451062316, 0.8953684771979732, 0.8927015734731976, 0.8917237087741132, 0.8895012890034669, 0.8890568050493377, 0.8866565916970397, 0.8857676237887813, 0.8852342430438261, 0.8845230687172193, 0.8837229975997867, 0.8828340296915281, 0.8814116810383145, 0.881144990665837, 0.8806116099208818, 0.879900435594275, 0.8791003644768424, 0.8789225708951907, 0.8780336029869322, 0.8776780158236288, 0.8768779447061961, 0.8765223575428926, 0.8759000800071117, 0.8755444928438083, 0.8747444217263757, 0.8741221441905948, 0.873944350608943, 0.8734109698639879, 0.8731442794915104, 0.8726997955373811, 0.8722553115832519, 0.871544137256645, 0.8707440661392124, 0.8705662725575607, 0.8700328918126056, 0.8696773046493022, 0.8675437816694818, 0.8669215041337007, 0.8656769490621389, 0.8659436394346164, 0.8646101875722286, 0.8638990132456218, 0.8623877678015823, 0.862565561383234, 0.8608765223575429, 0.8608765223575429, 0.8598097608676327, 0.8593652769135034, 0.8582985154235933, 0.8582096186327673, 0.8561649924437728, 0.8565205796070762, 0.8551871277446884, 0.8554538181171659, 0.853675882300649, 0.8533202951373455, 0.8519868432749578, 0.851897946484132, 0.8502089074584408, 0.8503867010400925, 0.8483420748510979, 0.848253178060272, 0.8455862743354965, 0.8455862743354965, 0.8441639256822828, 0.8445195128455862, 0.8437194417281536, 0.8439861321006311, 0.8425637834474176, 0.8423859898657658, 0.8413192283758556, 0.8408747444217264, 0.8403413636767713, 0.8401635700951195, 0.8398079829318161, 0.8397190861409903, 0.8386523246510801, 0.8389190150235577, 0.8388301182327318, 0.8389190150235577, 0.8366076984620855, 0.8370521824162148, 0.8366965952529114, 0.8371410792070406, 0.8373188727886923, 0.8374077695795181, 0.8373188727886923, 0.8374077695795181, 0.8375855631611698, 0.837496666370344, 0.8375855631611698, 0.8376744599519957, 0.8373188727886923, 0.8377633567428215, 0.8372299759978665, 0.8379411503244732, 0.8368743888345631, 0.8368743888345631, 0.8360743177171304, 0.8362521112987821, 0.8350075562272202, 0.8353631433905236, 0.8342963819006134, 0.8353631433905236, 0.835096453018046, 0.8352742465996977, 0.8341185883189617, 0.8349186594363943, 0.8347408658547426, 0.8348297626455685, 0.8349186594363943, 0.8353631433905236, 0.8349186594363943, 0.8349186594363943, 0.8349186594363943, 0.835096453018046, 0.8352742465996977, 0.8355409369721754, 0.8355409369721754, 0.8351853498088719, 0.8351853498088719, 0.8353631433905236, 0.8352742465996977, 0.8353631433905236, 0.8349186594363943, 0.835096453018046, 0.8352742465996977, 0.8349186594363943, 0.8345630722730909, 0.8344741754822651, 0.8337630011556583, 0.8337630011556583, 0.832696239665748, 0.8328740332473997, 0.8326073428749222, 0.8323406525024447, 0.8322517557116188, 0.8320739621299671, 0.8318072717574896, 0.8318961685483154, 0.8324295492932705, 0.8326073428749222, 0.8322517557116188, 0.8327851364565739, 0.8311849942217086, 0.831540581385012, 0.8294959551960174, 0.8300293359409725, 0.8286958840785847, 0.8287847808694107, 0.8266512578895902, 0.8271846386345453, 0.8247844252822473, 0.8254955996088541, 0.8222953151391235, 0.8227397990932528, 0.8193617210418704, 0.8197173082051737]
In [105]:
# plot the results (Manhattan distance)
plt.plot(k_potentials, grid_mean_scores)
plt.xlabel('Potential K values for KNN')
plt.ylabel('Cross-Validated Accuracy')
Out[105]:
<matplotlib.text.Text at 0x120cb5a50>

In [106]:
# plot the results (using Euclidean distance)
plt.plot(k_potentials, grid_mean_scores)
plt.xlabel('Potential K values for KNN')
plt.ylabel('Cross-Validated Accuracy')
Out[106]:
<matplotlib.text.Text at 0x12143b7d0>

In [107]:
# examine the best model
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
0.962396657481
{'n_neighbors': 1}
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
metric_params=None, n_jobs=1, n_neighbors=1, p=2,
weights='uniform')
In [ ]:

In [108]:
# read in the data & create matrices
X = X_scaled[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", \
"time_spend_company", "Work_accident", "promotion_last_5years", "low", "medium",\
"technical", "support", "IT", "product_mng", "marketing", "RandD", "accounting",\
"hr"]]
y = X_scaled["left"]
In [176]:
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
Support Vector Machines
In [177]:
# Train, fit and predict with SVM

from sklearn import svm
Model=svm.SVC(kernel='linear')
Model.fit(X_train,y_train)
Y_pred=Model.predict(X_test)
In [178]:
# Cross validation score for SVM
scores = cross_val_score(Model, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.780694712068
In [179]:
# Evaluate SVM for accuracy

from sklearn import metrics
metrics.accuracy_score(y_test,Y_pred)
Out[179]:
0.78106666666666669
In [180]:
# Confusion matrix for SVM

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2696  166]
[ 655  233]]
In [181]:
# Classification Accuracy confirmation between manual calculation and formula for SVM

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.781066666667
0.781066666667
In [182]:
# Misclassification Accuracy confirmation between manual calculation and formula for SVM
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.218933333333
0.218933333333
In [183]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated SVM

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.262387387387
0.262387387387
In [184]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for SVM

print(TN / float(TN + FP))
0.941998602376
In [185]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for SVM

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.583959899749
0.583959899749
In [186]:
# Misclassification Accuracy confirmation between manual calculation and formula for SVM
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.218933333333
0.218933333333
In [187]:
# Roc_curve for Support Vector Machine Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [92]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [93]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Support Vector Machines: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

Random Forest
In [157]:
# Random Forests Classifier from Scikit Learn

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_jobs=2)
RF.fit(X_train,y_train)
Y_pred=RF.predict(X_test)
In [158]:
# Score for 10-fold Cross Validation for Random Forests Classifier

scores = cross_val_score(RF, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.986309602501
In [159]:
# Prediction Accuracy for Random Forests

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.9896
In [160]:
# Confusion matrix for Random Forests

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2857    5]
[  34  854]]
In [161]:
# Classification Accuracy confirmation between manual calculation and formula for Random Forest

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.9896
0.9896
In [162]:
# Misclassification Accuracy confirmation between manual calculation and formula for Random Forest
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0104
0.0104
In [163]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for Random Forest Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.961711711712
0.961711711712
In [164]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for Random Forest Classifier

print(TN / float(TN + FP))
0.998252969951
In [165]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for Random Forest Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.994179278231
0.994179278231
In [166]:
# Misclassification Accuracy confirmation between manual calculation and formula for Random Forest
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0104
0.0104
In [167]:
# Roc_curve for Random Forest Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [168]:
y_pred_prob_Random_Forest = RF.predict_proba(X_test)[:, 1]
In [172]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_Random_Forest, bins=5)
plt.xlim(0, 1)
plt.title('Random Forest Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[172]:
<matplotlib.text.Text at 0x12013a8d0>

In [170]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [171]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Random Forest: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

Bagging
In [109]:
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
In [110]:
# Bagging Classifier

seed = 7
cart = DecisionTreeClassifier()
num_trees = 100
Bagging = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
Bagging.fit(X_train,y_train)
Y_pred=Bagging.predict(X_test)
In [111]:
# Cross Validation for Bagging

scores = cross_val_score(Bagging, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.987910076856
In [112]:
# Prediction Accuracy for Bagging

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.989333333333
In [113]:
# Confusion matrix for Bagging

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2854    8]
[  32  856]]
In [114]:
# Classification Accuracy confirmation between manual calculation and formula for Bagging

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.989333333333
0.989333333333
In [115]:
# Misclassification Accuracy confirmation between manual calculation and formula for Bagging
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0106666666667
0.0106666666667
In [116]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for Bagging Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.963963963964
0.963963963964
In [117]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for Bagging Classifier

print(TN / float(TN + FP))
0.997204751922
In [118]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for Bagging Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.990740740741
0.990740740741
In [119]:
# Misclassification Accuracy confirmation between manual calculation and formula for Bagging
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0106666666667
0.0106666666667
In [120]:
# Roc_curve for Bagging


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [121]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
In [122]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Bagging: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [123]:
y_pred_prob_Bagging = Bagging.predict_proba(X_test)[:, 1]
In [124]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_Bagging, bins=5)
plt.xlim(0, 1)
plt.title('Bagging Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[124]:
<matplotlib.text.Text at 0x11b217610>

In [ ]:

AdaBoost
In [125]:
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
In [126]:
# AdaBoost Classifier

seed = 7
cart = AdaBoostClassifier()
num_trees = 100
AdaBoostClassifier = AdaBoostClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
AdaBoostClassifier.fit(X_train,y_train)
Y_pred=AdaBoostClassifier.predict(X_test)
In [127]:
# Cross Validation for AdaBoost

scores = cross_val_score(AdaBoostClassifier, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.958576016468
In [128]:
# Prediction Accuracy for AdaBoost

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.958133333333
In [129]:
# Confusion matrix for AdaBoost

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2793   69]
[  88  800]]
In [130]:
# Classification Accuracy confirmation between manual calculation and formula for Bagging

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.958133333333
0.958133333333
In [131]:
# Misclassification Accuracy confirmation between manual calculation and formula for AdaBoost
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0418666666667
0.0418666666667
In [132]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for AdaBoost Boosting Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.900900900901
0.900900900901
In [133]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for AdaBoost Classifier

print(TN / float(TN + FP))
0.975890985325
In [134]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for AdaBoost Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.920598388953
0.920598388953
In [135]:
# Misclassification Accuracy confirmation between manual calculation and formula for AdaBoost
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0418666666667
0.0418666666667
In [136]:
# Roc_curve for AdaBoost


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [137]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
In [138]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='AdaBoost: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [139]:
y_pred_prob_AdaBoost = AdaBoostClassifier.predict_proba(X_test)[:, 1]
In [140]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_AdaBoost, bins=5)
plt.xlim(0, 1)
plt.title('AdaBoost Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[140]:
<matplotlib.text.Text at 0x11b367a10>

In [ ]:

In [ ]:

In [ ]:

Gradient Boosting
In [141]:
from sklearn.ensemble import GradientBoostingClassifier
In [142]:
# Train and predict Gradient Boosting

GradientBoosting = GradientBoostingClassifier()
GradientBoosting.fit(X_train,y_train)
Y_pred=GradientBoosting.predict(X_test)
In [143]:
# Cross Validation for Gradient Boosting

scores = cross_val_score(GradientBoosting, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.976443246546
In [144]:
# Prediction Accuracy for Gradient Boosting

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.974133333333
In [145]:
# Confusion matrix for Gradient Boosting

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2832   30]
[  67  821]]
In [146]:
# Classification Accuracy confirmation between manual calculation and formula for Gradient Boosting

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.974133333333
0.974133333333
In [147]:
# Misclassification Accuracy confirmation between manual calculation and formula for Gradient Boosting
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0258666666667
0.0258666666667
In [148]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for Gradient Boosting Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.92454954955
0.92454954955
In [149]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for Gradient Classifier

print(TN / float(TN + FP))
0.989517819706
In [150]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for Gradient Boosting Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.964747356052
0.964747356052
In [151]:
# Misclassification Accuracy confirmation between manual calculation and formula for Gradient Boosting
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0258666666667
0.0258666666667
In [152]:
# Roc_curve for Gradient Boosting


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Gradient Boosting Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [153]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
In [154]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Gradient Boosting: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [155]:
y_pred_prob_GradientBoosting = GradientBoosting.predict_proba(X_test)[:, 1]
In [156]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_GradientBoosting, bins=5)
plt.xlim(0, 1)
plt.title('Gradient Boosting Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[156]:
<matplotlib.text.Text at 0x11c2566d0>

In [ ]:

In [ ]:

Part IV
Conclusion
One interesting chart showed us which departments have the most people leaving (and staying). They are Sales followed by Technical and Support. That means that the biggest effort at the company should be placed on keeping the strong employees in those departments (but by no means neglect the other departments).
There are several other interesting patterns in the data. Some are very intuitive whereas others very surprising and unexpected. There are three clusters of people leaving the company (the most important cluster is not as dense, in the sense that there are actually slightly more people there that stay than leave, but nevertheless its the most interesting and surprising cluster).
The three groups are employees with high performance and low satisfaction, employees with low satisfcation and performance (rather obvious) and employees with high satisfaction and performance (a surprising result). The first two results are very strong but rather trivial, there is limited use in providing detailed analyses of them. However, the somewhat less dense but very significant group of highly satisfied employees that do remarkably well in their evaluations is of extreme importance. Why do they leave? Well, there are several important factors to consider.
The people with high salaries and at even average satisfaction levels rarely leave the firm. Hence, one attribute that people who tend to leave will generally possess is salaries that are not high and fall in the low or medium range. It is interesting to note though that people with near perfect to perfect satisfaction levels do not leave even if they are in the low or medium income range brackets. Hence, a possible course of action to potentially alleviate this issue is increase salaries for good employees. Also, a further investigation into what might be the differentiating factor between people who are very satisfied (0.7-0.9 rate) vs super satisfied (0.9-1.0) can be done to see if it is possible to further increase the satisfaction level of those that are very satisfied but not perfectly so (maybe its possible to do this while at the same time saving money by not instituting as much pay increases if there is a way to get them satisfied in another manner, perhaps by working less hours for instance) would be a good idea.
Secondly, promotions are a huge factor. Looking at the plot of promotions during the past five years, it is very clear that people who are promoted stay and people who are not promoted leave. Of course, promotions and salary increases have some correlation because people tend to receive salary increases during promotions. Hence, it's important to promote employees if they do good work! That will help decrease the turnover rate.
The third important point is that employees who work a lot of hours that stay at the company for at least a medium amount of time tend to leave. Only the relatively small number of employees who work at the company for many years stay at the company regardless of working hours but very few make it there. Hence, the hours worked plays an important decision making role when employees think about whether to stay at the firm or seek other employment. Due to this finding it is important to see if any kind of rebalancing can be done so that part of the work can perhaps be shifted between employees who do not work as many hours and those that do in order to make the distribution of hours worked by employees more uniform which can perhaps increase employees willingness to stay at the company longer. It is quiet possible that some employees just feel burnt out even if they are generally happy with their job. Due to an inefficient work life balance, they might seek to find employment elsewhere.
Multiple machine learning algorithms were ran on the data and quiet a few did relatively well in terms of making predictions on out of sample data as measured by the accuracy score, the confusion matrix and deriatives thereof as well as the ROC Curve. In particular, the Random Forest and Bagging algorithms did the best with accuracy and cross-validation scores marginally different from each other just under 99%. Both correctly predicted over 3,700 out of 3,750 employees accurately (compared to a baseline accuracy of only 76% (2,850) as to whether they would leave/stay). The various scores computed from the confusion matrix were all almost perfect. Particularly importantly, the ROC curve has an area covering of .98 for both those algorithms.
Hence, using either of these two algorithms (Random Forests or Bagging) have given us great predictions as to which employees will leave vs which ones would stay.
Assumptions
We have assumed the dataset is accurate as we have not audited its validity.
Another assumption we made is that we trained on 75% of the data, conducted cross-validation, and then tested on the 25% remaining out of sample. However, when trying to sample to train a larger sample of the data (such as 90/10 test/train split), we received very similar results in our algorithms. Hence, our model is robust as should perform well when used on future data.
An assumption made by the Random Forest Model which did exceptionally well is that the sampling that it conducted during boostrap aggregation is actually representative. This is a safe assumption to make and the model has done really well on out of sample data.
Introduction:
This project analyzes a dataset in an attempt to determine what causes the good employees to leave their current jobs. The first step is to clean the data (Part I) and visualize it (Part II) using various types of graphs and charts to help establish clear relationships between various features. Part III analyzes the dataset using various Machine Learning algorithms and picks the best algorithm to model this dataset based on various metrics. Lastly, a conclusion on the entire analysis is provided.
Part I
Data Cleaning
This initial stage of the project is to clean the data in such a way that all the important attributes of the dataset be retained and unhindered for analysis purposes. Hence, the initial cleaning phase of the project imports the dataset and the libraries needed for the cleaning stages of the project. A check for Null Values/Missing Values is conducted (none are present). The multiple categorical variables in the dataset are created into dummy variables which is necessary for later stages of the analysis. These new variables are afterwards added to the original dataset and the initial variables from which dummification was done are removed to avoid redundancy. Lastly, the dataset is normalized due to the need to apply machine learning algorithms to it in later stages of the project.
In [1]:
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

%matplotlib inline
In [2]:
HR = pd.read_csv('HR_comma_sep.csv')
HR.head(10)
Out[2]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary
0	0.38	0.53	2	157	3	0	1	0	sales	low
1	0.80	0.86	5	262	6	0	1	0	sales	medium
2	0.11	0.88	7	272	4	0	1	0	sales	medium
3	0.72	0.87	5	223	5	0	1	0	sales	low
4	0.37	0.52	2	159	3	0	1	0	sales	low
5	0.41	0.50	2	153	3	0	1	0	sales	low
6	0.10	0.77	6	247	4	0	1	0	sales	low
7	0.92	0.85	5	259	5	0	1	0	sales	low
8	0.89	1.00	5	224	5	0	1	0	sales	low
9	0.42	0.53	2	142	3	0	1	0	sales	low
In [3]:
HR[HR['satisfaction_level'] == np.nan]
HR[HR['last_evaluation'] == np.nan]
HR[HR['number_project'] == np.nan]
HR[HR['average_montly_hours'] == np.nan]
HR[HR['time_spend_company'] == np.nan]
HR[HR['Work_accident'] == np.nan]
HR[HR['left'] == np.nan]
HR[HR['sales'] == np.nan]
HR[HR['salary'] == np.nan]
Out[3]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary
In [4]:
# Create a heatmap, to check for null values.

plt.figure(figsize=(10,7))
sns.heatmap(HR.isnull(),yticklabels=False, cbar=False, cmap='viridis')
Out[4]:
<matplotlib.axes._subplots.AxesSubplot at 0x114c6ee50>

In [5]:
HR['sales'].value_counts()
Out[5]:
sales          4140
technical      2720
support        2229
IT             1227
product_mng     902
marketing       858
RandD           787
accounting      767
hr              739
management      630
Name: sales, dtype: int64
In [6]:
HR['salary'].value_counts()
Out[6]:
low       7316
medium    6446
high      1237
Name: salary, dtype: int64
In [7]:
HR['left'].value_counts()
Out[7]:
0    11428
1     3571
Name: left, dtype: int64
In [8]:
# Create dummy variables for all the departments

sales = pd.get_dummies(HR.sales)

sales.columns = ["IT", "RandD", "accounting", "hr", "management", "marketing", "product_mng", "sales_department", "support", "technical"]

sales.head()
Out[8]:
IT	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical
0	0	0	0	0	0	0	0	1	0	0
1	0	0	0	0	0	0	0	1	0	0
2	0	0	0	0	0	0	0	1	0	0
3	0	0	0	0	0	0	0	1	0	0
4	0	0	0	0	0	0	0	1	0	0
In [9]:
# Create dummy variables for the salary

salary = pd.get_dummies(HR.salary)

salary.columns = ['low', 'medium', 'high']

salary.head()
Out[9]:
low	medium	high
0	0	1	0
1	0	0	1
2	0	0	1
3	0	1	0
4	0	1	0
In [10]:
# Create dummy variable for stayed/left

left = pd.get_dummies(HR.left)

left.columns = ['left', 'stayed']

left.head(10)
Out[10]:
left	stayed
0	0	1
1	0	1
2	0	1
3	0	1
4	0	1
5	0	1
6	0	1
7	0	1
8	0	1
9	0	1
In [11]:
# Concatenate the original dataset with the newly created salary and sales dummified variables

HR = pd.concat([HR, salary, sales], axis = 1)

HR.head(10)
Out[11]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary	...	IT	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical
0	0.38	0.53	2	157	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
1	0.80	0.86	5	262	6	0	1	0	sales	medium	...	0	0	0	0	0	0	0	1	0	0
2	0.11	0.88	7	272	4	0	1	0	sales	medium	...	0	0	0	0	0	0	0	1	0	0
3	0.72	0.87	5	223	5	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
4	0.37	0.52	2	159	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
5	0.41	0.50	2	153	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
6	0.10	0.77	6	247	4	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
7	0.92	0.85	5	259	5	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
8	0.89	1.00	5	224	5	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
9	0.42	0.53	2	142	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
10 rows × 23 columns
In [12]:
HR.dtypes
Out[12]:
satisfaction_level       float64
last_evaluation          float64
number_project             int64
average_montly_hours       int64
time_spend_company         int64
Work_accident              int64
left                       int64
promotion_last_5years      int64
sales                     object
salary                    object
low                        uint8
medium                     uint8
high                       uint8
IT                         uint8
RandD                      uint8
accounting                 uint8
hr                         uint8
management                 uint8
marketing                  uint8
product_mng                uint8
sales_department           uint8
support                    uint8
technical                  uint8
dtype: object
In [13]:
# Create a new column 

salary_map = {'low': 1, 'medium': 2, 'high': 3}

HR['salary_variable'] = HR['salary'].apply(lambda x: salary_map[x])

HR.head()
Out[13]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary	...	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical	salary_variable
0	0.38	0.53	2	157	3	0	1	0	sales	low	...	0	0	0	0	0	0	1	0	0	1
1	0.80	0.86	5	262	6	0	1	0	sales	medium	...	0	0	0	0	0	0	1	0	0	2
2	0.11	0.88	7	272	4	0	1	0	sales	medium	...	0	0	0	0	0	0	1	0	0	2
3	0.72	0.87	5	223	5	0	1	0	sales	low	...	0	0	0	0	0	0	1	0	0	1
4	0.37	0.52	2	159	3	0	1	0	sales	low	...	0	0	0	0	0	0	1	0	0	1
5 rows × 24 columns
In [14]:
del HR['salary']

del HR['sales']
In [15]:
HR.head()
Out[15]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	low	medium	...	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical	salary_variable
0	0.38	0.53	2	157	3	0	1	0	0	1	...	0	0	0	0	0	0	1	0	0	1
1	0.80	0.86	5	262	6	0	1	0	0	0	...	0	0	0	0	0	0	1	0	0	2
2	0.11	0.88	7	272	4	0	1	0	0	0	...	0	0	0	0	0	0	1	0	0	2
3	0.72	0.87	5	223	5	0	1	0	0	1	...	0	0	0	0	0	0	1	0	0	1
4	0.37	0.52	2	159	3	0	1	0	0	1	...	0	0	0	0	0	0	1	0	0	1
5 rows × 22 columns
In [16]:
HR.dtypes
Out[16]:
satisfaction_level       float64
last_evaluation          float64
number_project             int64
average_montly_hours       int64
time_spend_company         int64
Work_accident              int64
left                       int64
promotion_last_5years      int64
low                        uint8
medium                     uint8
high                       uint8
IT                         uint8
RandD                      uint8
accounting                 uint8
hr                         uint8
management                 uint8
marketing                  uint8
product_mng                uint8
sales_department           uint8
support                    uint8
technical                  uint8
salary_variable            int64
dtype: object
In [17]:
from sklearn.preprocessing import MinMaxScaler
X_scaled = MinMaxScaler().fit_transform(HR.astype(float))
X_scaled = pd.DataFrame(X_scaled, columns = HR.columns)

# Min Max Scaler

X_scaled.head(10)
Out[17]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	low	medium	...	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical	salary_variable
0	0.318681	0.265625	0.0	0.285047	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
1	0.780220	0.781250	0.6	0.775701	0.500	0.0	1.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.5
2	0.021978	0.812500	1.0	0.822430	0.250	0.0	1.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.5
3	0.692308	0.796875	0.6	0.593458	0.375	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
4	0.307692	0.250000	0.0	0.294393	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
5	0.351648	0.218750	0.0	0.266355	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
6	0.010989	0.640625	0.8	0.705607	0.250	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
7	0.912088	0.765625	0.6	0.761682	0.375	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
8	0.879121	1.000000	0.6	0.598131	0.375	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
9	0.362637	0.265625	0.0	0.214953	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
10 rows × 22 columns
Part II
Exploratory Data Analysis (EDA)
In this section, we try to visualize the data on employees using various plots, graphs, charts in order to get an idea of how different features in the dataset relate to each other.
The below represents a heatmap of correlations of features. Some interesting observations from this heatmap: Quiet intuitively, you can see that the relationship between satisfaction level and people that left the company is negative. It's also interesting to note that there is a positive correlation between the number of hours employees work and their evaluation. These and other relationships will be explored more deeply in the rest of the EDA.
In [18]:
correlation = X_scaled.corr()
plt.figure(figsize=(20,20))
sns.heatmap(correlation, vmax=10, square=True,annot=True,cmap=None, linewidths=1)

plt.title('Correlation between features')
Out[18]:
<matplotlib.text.Text at 0x118f3a050>

In [19]:
# Exported csv file to work with in Tableau
X_scaled.to_csv("Updated_Capstone.csv")
In [20]:
X_scaled['left'].value_counts()
Out[20]:
0.0    11428
1.0     3571
Name: left, dtype: int64
In [21]:
stayed = np.where(X_scaled['left'] == 0)[0]
left = np.where(X_scaled['left'] == 1)[0]
For a given level of satisfaction, there is definitely an intuitive pattern that the higher someone's salary is, the less they are likely to leave. This relationship is a lot more apparent in the distinction between the high income earners vs low & median income earners as a group. The difference in the people that left between low and median income salaries is not significant. Hence, most of the people that leave are not receiving a high salary, which means that you if you would like to retain these employees, you should increase their pay!
In [22]:
# plot of Satisfaction vs Last Evaluation
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.satisfaction_level[stayed], X_scaled.salary_variable[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.satisfaction_level[left], X_scaled.salary_variable[left], alpha=0.1, color='r')
ax.set_title("satisfaction_level vs salary")
ax.set_xlabel("satisfaction_level")
ax.set_ylabel("salary")
plt.show()

In the Satisfaction vs Last Evaluation plot below, we can note some interesting observations. Firstly, there is a sizeable cluster of people with strong evaluations and low satisfaction levels leaving. Secondly, there is a significant group of people leaving that are slighly below average in performance and evaluation. Lastly, and by far the most interesting observation of the three, there is a decent size of a not very dense cluster of people leaving that have done extremely well on their evaluations (0.7 to 1 on a 0-1 scale) and and have very high satisfaction rates, albeit not perfect (0.7-0.9 on a 0-1 scale). That particular group of employees also have many people that stay, but a significant proportion of that group is leaving. So the most surprising thing to observe is that people can be satisfied with the job and still leave! There are several possibilites for why this might be possible. People might enjoy their job but think they can do even better elsewhere, whether its income or career growth (or a combination of the two, considering that those things tend to go hand in hand). Another possibility is that people might not be honest in surveys because of fear of being reprimanded for negative surveys.
In [23]:
# plot of Satisfaction vs Last Evaluation
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.satisfaction_level[stayed], X_scaled.last_evaluation[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.satisfaction_level[left], X_scaled.last_evaluation[left], alpha=0.1, color='r')
ax.set_title("satisfaction_level vs last_evaluation")
ax.set_xlabel("satisfaction_level")
ax.set_ylabel("last_evaluation")
plt.show()

The Satisfaction vs Promotion during the last five years plot below shows a very strong relationship in several interesting ways! People who have been promoted don't leave very often! However, such promotions are not common enough! Therefore, if you don't want good employees to leave, promoting them will really help!
In [24]:
# plot of Satisfaction vs Promotion during last five years
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.satisfaction_level[stayed], X_scaled.promotion_last_5years[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.satisfaction_level[left], X_scaled.promotion_last_5years[left], alpha=0.1, color='r')
ax.set_title("satisfaction_level vs promotion_last_5years")
ax.set_xlabel("satisfaction_level")
ax.set_ylabel("promotion_last_5years")
plt.show()

The below scatter plot compares the average working hours against the time spent at the company and how it impacts the employee turnover rate. There are multiple interesting relationships to note. For employees that just started out, the majority tend to stick around for some time even if they work substantial hours. However, for employees who have been at the company for a while, there is a significant tendency to leave the company if working substantial hours. Consequently, we can observe that very few employees stick around at the company for a long time period because the ones who were working long hours almost all left! However, those that do stick around for a long time (relatively few) tend to all be devoted and not leave
There are several conjectures for why this is possible. One is that employees have worked for a significant amount of hours and put in a lot of time and effort expecting a quick promotion and/or pay increase (refer to graph above for relationship between promotions and staying/leaving) that they have not received and leave for elsewhere. Another reason might be that people have tried working for a little bit of time, realized that they are being overworked and have started looking for other opportunities to seek employment.
In [25]:
# plot of Average_monthly_hours vs time_spend_company
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.average_montly_hours[stayed], X_scaled.time_spend_company[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.average_montly_hours[left], X_scaled.time_spend_company[left], alpha=0.1, color='r')
ax.set_title("average_monthly_hours vs time_spend_company")
ax.set_xlabel("average_monthly_hours")
ax.set_ylabel("time_spend_company")
plt.show()

In [26]:
# Imported from Tableau
from IPython.display import Image
Image(filename = 'Average Monthly Hours vs Satisfaction Level.png', width = 1000)
Out[26]:

There does not appear to be a strong relationship between accidents and people leaving the company, hence, people stay/leave regardless of the accidents they might have experienced.
In [27]:
# plot of work_accident vs time_spend_company
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.Work_accident[stayed], X_scaled.time_spend_company[stayed], alpha=0.5, color='g')
ax.scatter(X_scaled.Work_accident[left], X_scaled.time_spend_company[left], alpha=0.5, color='r')
ax.set_title("Work_accident vs time_spend_company")
ax.set_xlabel("Work_accident")
ax.set_ylabel("time_spend_company")
plt.show()

The below stripplot compares number of projects done by an employee versus the number of promotions they have received over the past 5 years. It is interesting to note that the number of projects completed has very insignificant impact on whether a person leaves. On the contrary, the promotions received during the past 5 years is of critical importance. The majority of people who have received a promotion stay. Almost all those who were not promoted leave.
The data seems insufficient to be able to determine why the number of projects is not related to promotions. There are at least several possibilities. Perhaps the number of projects refers to the number assigned (vs completed) or that the projects are not all of equal difficulty and not equally time consuming leading to distorted hypothesis that the number of projects might necessarily correspond to an employee's productivity.
In [28]:
import seaborn as sns

sns.stripplot("number_project", "promotion_last_5years", data=X_scaled, hue="left", jitter = True)
Out[28]:
<matplotlib.axes._subplots.AxesSubplot at 0x117949d90>

The Satisfaction level distribution below shows that the majority of people are at least moderately satisfied with their job. Therefore, for the people that are satisfied and do well, efforts should be placed on making them even more happy by promoting them. For the people that do poorly or mediocre, either studies should be placed on determining why they are not doing well and trying to improve their performance or attempting to replace them with better employees (this would depend on a cost/benefit analysis) done by the company regarding this issue.
In [29]:
# Histogram distribution of Satisfaction Level

f,ax1 = plt.subplots(1,1)
sns.distplot(X_scaled['satisfaction_level'],bins=10,norm_hist=False)
plt.show()

The distribution (which is a decent approximation of a normal distribution with almost no tails and some kurtosis present) of the last evaluation graph shows us there is a sizeable portion of people that do well in their evaluations, hence the company should try to do as much as possible to keep those people!
In [30]:
# Histogram distribution of Last Evaluation


f,ax1 = plt.subplots(1,1)
sns.distplot(X_scaled['last_evaluation'],bins=10,norm_hist=False)
plt.show()

The below plot shows the ratios of people who stay in the company by department and the ratio of people who leave the company by department. It is clear from the chart that the departments with the most number of employees also have the highest ratios of both people who have left and those that stayed at the company. This tells us there should be particular effort place on retaining good employees that work in those departments, in particular the sales, technical and support departments, respectively. The largest efforts should be placed on retaining the good employees in these groups because they have the most people leaving.
In [34]:
# This plot is to compare the people that left across different departments
# The non-department variables are deleted to only keep the departments

del X_scaled['satisfaction_level']
del X_scaled['last_evaluation']
del X_scaled['number_project']
del X_scaled['average_montly_hours']
del X_scaled['time_spend_company']
del X_scaled['Work_accident']
del X_scaled['promotion_last_5years']
del X_scaled['low']
del X_scaled['medium']
del X_scaled['high']
del X_scaled['salary_variable']


left=X_scaled.groupby('left').mean()
left=left.transpose()
plt.figure(figsize=(20,10))
left.plot(kind='bar')
Out[34]:
<matplotlib.axes._subplots.AxesSubplot at 0x11a7bcd10>
<matplotlib.figure.Figure at 0x11a360850>

Part III
Predictive Modelling
In this section below, multiple models will be run that are applicable to classification. Afterwards, various scoring metrics will be calculated for the models and a determination will be made as to which model is based for this dataset.
In [174]:
# read in the data & create matrices
X = X_scaled[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", \
"time_spend_company", "Work_accident", "promotion_last_5years", "salary_variable",\
"technical", "support", "IT", "product_mng", "marketing", "RandD", "accounting",\
"hr", "management", "sales_department"]]
y = X_scaled["left"]
In [175]:
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
Logistic Regression Model
In [33]:
# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
model = LogisticRegression()

# fit the model with data
mdl = model.fit(X_train, y_train)

# predict the response for new observations

logit = model.predict(X_test)

# sklearn output to check that ran on right data split.

len(logit)
Out[33]:
3750
In [34]:
# These represent the y-intercept and coefficients for all the variables in the logistic regression.

print(model.intercept_)
print(model.coef_)
[ 0.57305398]
[[-3.71487625  0.45484967 -1.50619896  0.8704181   1.97590531 -1.45425383
-1.26024249 -1.32512177  0.22588571  0.23497924 -0.01282827  0.07509077
0.13925062 -0.47542127  0.26606552  0.3920348  -0.44451555  0.17251241]]
In [35]:
# Prediction Accuracy for Logistic Regression

from sklearn import metrics
print(metrics.accuracy_score(y_test, logit))
0.796
In [36]:
from sklearn.cross_validation import cross_val_score
In [37]:
# 10-fold cross-validation for Logistic Regression to find the CV score
mdl = model.fit(X_train, y_train)
scores = cross_val_score(mdl, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.794205514996
In [38]:
from sklearn import metrics
confusion = metrics.confusion_matrix(y_test, logit)
print ((confusion))
[[2651  211]
[ 554  334]]
In [39]:
y_pred_prob = model.predict_proba(X_test)[:, 1]
In [40]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob, bins=5)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[40]:
<matplotlib.text.Text at 0x1181a6a90>

In [41]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [42]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, logit)


plot_confusion_matrix(cm, title='LogisticRegression: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [43]:
# Roc_curve for Logistic Regression Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, logit)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Logistic Regression Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

K Nearest Nearbours Model
K=1
In [44]:
from sklearn.neighbors import KNeighborsClassifier
In [45]:
# K-Nearest Neighbours with K=1
knn = KNeighborsClassifier(n_neighbors=1, metric = 'euclidean')
In [46]:
y = knn.fit(X_train, y_train)

KNN = knn.predict(X_test)

# sklearn output to check that ran on right data split.

len(KNN)
Out[46]:
3750
In [47]:
# Cross fold validation for K=1

md2 = knn.fit(X_train, y_train)
scores = cross_val_score(md2, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.962043001151
In [48]:
# Prediction Accuracy.

from sklearn import metrics
print(metrics.accuracy_score(y_test, KNN))
0.967466666667
In [49]:
confusion = metrics.confusion_matrix(y_test, KNN)
print confusion
[[2775   87]
[  35  853]]
In [50]:
# Classification Accuracy confirmation between manual calculation and formula for K=1

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, KNN))
0.967466666667
0.967466666667
In [51]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=1
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN))
0.0325333333333
0.0325333333333
In [52]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for K=1

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, KNN))
0.960585585586
0.960585585586
In [53]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for K=1


print(TN / float(TN + FP))
0.969601677149
In [54]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for K=1

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, KNN))
0.907446808511
0.907446808511
In [55]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=1
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN))
0.0325333333333
0.0325333333333
In [56]:
y_pred_prob_knn = knn.predict_proba(X_test)[:, 1]
In [57]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_knn, bins=5)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[57]:
<matplotlib.text.Text at 0x1181d6b90>

In [58]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [59]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, KNN)


plot_confusion_matrix(cm, title='KNN for K=1: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [60]:
# Roc_curve for KNN=1 Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, KNN)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('KNN=1 Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

K=5
In [61]:
# K-Nearest Neighbours with K=5

knn = KNeighborsClassifier(n_neighbors=5, metric = 'euclidean')
y = knn.fit(X_train, y_train)
KNN5 = knn.predict(X_test)

# sklearn output to check that ran on right data split.

len(KNN5)
Out[61]:
3750
In [62]:
# Cross Validation for K=5 Cross-Fold Validation

md2 = knn.fit(X_train, y_train)
scores = cross_val_score(md2, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.938309179837
In [63]:
# Prediction Accuracy on KNN=5 for accuracy score

from sklearn import metrics
print(metrics.accuracy_score(y_test, KNN5))
0.941333333333
In [64]:
confusion = metrics.confusion_matrix(y_test, KNN5)
print confusion
[[2731  131]
[  89  799]]
In [65]:
# Classification Accuracy confirmation between manual calculation and formula for K=5

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, KNN5))
0.941333333333
0.941333333333
In [66]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=5
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN5))
0.0586666666667
0.0586666666667
In [67]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for K=5

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, KNN5))
0.899774774775
0.899774774775
In [68]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for K=5

print(TN / float(TN + FP))
0.954227812718
In [69]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for K=5

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, KNN5))
0.859139784946
0.859139784946
In [70]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=5
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN5))
0.0586666666667
0.0586666666667
In [71]:
# Roc_curve for KNN=5 Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, KNN5)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [72]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [73]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, KNN5)


plot_confusion_matrix(cm, title='KNN for N=5: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [74]:
y_pred_prob_knn = knn.predict_proba(X_test)[:, 1]
In [75]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_knn, bins=5)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[75]:
<matplotlib.text.Text at 0x11acdea10>

In [ ]:

Grid Search for optimal N value in KNN
In [101]:
from sklearn.grid_search import GridSearchCV
/Users/Misha/anaconda/lib/python2.7/site-packages/sklearn/grid_search.py:43: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.
DeprecationWarning)
In [102]:
k_potentials = list(range(1, 200))
print(k_potentials)
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]
In [103]:
# Set Nearest Neighbours equal to k
knn = KNeighborsClassifier(n_neighbors=k_potentials, metric = 'euclidean')
# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_potentials)
# instantiate the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
# fit the grid with data
grid.fit(X_train, y_train)
# view the results
grid.grid_scores_
Out[103]:
[mean: 0.96240, std: 0.00591, params: {'n_neighbors': 1},
mean: 0.95884, std: 0.00543, params: {'n_neighbors': 2},
mean: 0.94568, std: 0.00463, params: {'n_neighbors': 3},
mean: 0.94622, std: 0.00555, params: {'n_neighbors': 4},
mean: 0.93875, std: 0.00620, params: {'n_neighbors': 5},
mean: 0.94284, std: 0.00673, params: {'n_neighbors': 6},
mean: 0.93795, std: 0.00746, params: {'n_neighbors': 7},
mean: 0.93839, std: 0.00819, params: {'n_neighbors': 8},
mean: 0.93351, std: 0.00932, params: {'n_neighbors': 9},
mean: 0.93493, std: 0.00848, params: {'n_neighbors': 10},
mean: 0.93048, std: 0.00825, params: {'n_neighbors': 11},
mean: 0.93253, std: 0.00810, params: {'n_neighbors': 12},
mean: 0.92933, std: 0.00898, params: {'n_neighbors': 13},
mean: 0.92968, std: 0.00824, params: {'n_neighbors': 14},
mean: 0.92719, std: 0.00747, params: {'n_neighbors': 15},
mean: 0.92817, std: 0.00830, params: {'n_neighbors': 16},
mean: 0.92559, std: 0.00873, params: {'n_neighbors': 17},
mean: 0.92728, std: 0.00821, params: {'n_neighbors': 18},
mean: 0.92497, std: 0.00821, params: {'n_neighbors': 19},
mean: 0.92470, std: 0.00830, params: {'n_neighbors': 20},
mean: 0.92204, std: 0.00804, params: {'n_neighbors': 21},
mean: 0.92186, std: 0.00867, params: {'n_neighbors': 22},
mean: 0.91902, std: 0.00856, params: {'n_neighbors': 23},
mean: 0.91777, std: 0.00850, params: {'n_neighbors': 24},
mean: 0.91564, std: 0.00809, params: {'n_neighbors': 25},
mean: 0.91457, std: 0.00913, params: {'n_neighbors': 26},
mean: 0.91244, std: 0.00927, params: {'n_neighbors': 27},
mean: 0.91244, std: 0.00873, params: {'n_neighbors': 28},
mean: 0.91101, std: 0.00878, params: {'n_neighbors': 29},
mean: 0.91110, std: 0.00853, params: {'n_neighbors': 30},
mean: 0.90906, std: 0.00855, params: {'n_neighbors': 31},
mean: 0.90941, std: 0.00906, params: {'n_neighbors': 32},
mean: 0.90773, std: 0.00922, params: {'n_neighbors': 33},
mean: 0.90728, std: 0.00991, params: {'n_neighbors': 34},
mean: 0.90550, std: 0.00946, params: {'n_neighbors': 35},
mean: 0.90417, std: 0.00891, params: {'n_neighbors': 36},
mean: 0.90292, std: 0.00887, params: {'n_neighbors': 37},
mean: 0.90097, std: 0.00881, params: {'n_neighbors': 38},
mean: 0.90035, std: 0.00948, params: {'n_neighbors': 39},
mean: 0.89875, std: 0.01009, params: {'n_neighbors': 40},
mean: 0.89759, std: 0.01009, params: {'n_neighbors': 41},
mean: 0.89626, std: 0.00955, params: {'n_neighbors': 42},
mean: 0.89537, std: 0.01016, params: {'n_neighbors': 43},
mean: 0.89270, std: 0.01221, params: {'n_neighbors': 44},
mean: 0.89172, std: 0.01244, params: {'n_neighbors': 45},
mean: 0.88950, std: 0.01089, params: {'n_neighbors': 46},
mean: 0.88906, std: 0.01150, params: {'n_neighbors': 47},
mean: 0.88666, std: 0.01137, params: {'n_neighbors': 48},
mean: 0.88577, std: 0.01143, params: {'n_neighbors': 49},
mean: 0.88523, std: 0.01132, params: {'n_neighbors': 50},
mean: 0.88452, std: 0.01149, params: {'n_neighbors': 51},
mean: 0.88372, std: 0.01237, params: {'n_neighbors': 52},
mean: 0.88283, std: 0.01227, params: {'n_neighbors': 53},
mean: 0.88141, std: 0.01166, params: {'n_neighbors': 54},
mean: 0.88114, std: 0.01085, params: {'n_neighbors': 55},
mean: 0.88061, std: 0.01041, params: {'n_neighbors': 56},
mean: 0.87990, std: 0.01105, params: {'n_neighbors': 57},
mean: 0.87910, std: 0.00974, params: {'n_neighbors': 58},
mean: 0.87892, std: 0.00997, params: {'n_neighbors': 59},
mean: 0.87803, std: 0.01108, params: {'n_neighbors': 60},
mean: 0.87768, std: 0.01119, params: {'n_neighbors': 61},
mean: 0.87688, std: 0.01152, params: {'n_neighbors': 62},
mean: 0.87652, std: 0.01182, params: {'n_neighbors': 63},
mean: 0.87590, std: 0.01129, params: {'n_neighbors': 64},
mean: 0.87554, std: 0.01148, params: {'n_neighbors': 65},
mean: 0.87474, std: 0.01183, params: {'n_neighbors': 66},
mean: 0.87412, std: 0.01223, params: {'n_neighbors': 67},
mean: 0.87394, std: 0.01196, params: {'n_neighbors': 68},
mean: 0.87341, std: 0.01210, params: {'n_neighbors': 69},
mean: 0.87314, std: 0.01231, params: {'n_neighbors': 70},
mean: 0.87270, std: 0.01219, params: {'n_neighbors': 71},
mean: 0.87226, std: 0.01168, params: {'n_neighbors': 72},
mean: 0.87154, std: 0.01185, params: {'n_neighbors': 73},
mean: 0.87074, std: 0.01104, params: {'n_neighbors': 74},
mean: 0.87057, std: 0.01132, params: {'n_neighbors': 75},
mean: 0.87003, std: 0.01129, params: {'n_neighbors': 76},
mean: 0.86968, std: 0.01148, params: {'n_neighbors': 77},
mean: 0.86754, std: 0.01056, params: {'n_neighbors': 78},
mean: 0.86692, std: 0.01041, params: {'n_neighbors': 79},
mean: 0.86568, std: 0.01028, params: {'n_neighbors': 80},
mean: 0.86594, std: 0.01024, params: {'n_neighbors': 81},
mean: 0.86461, std: 0.00937, params: {'n_neighbors': 82},
mean: 0.86390, std: 0.00915, params: {'n_neighbors': 83},
mean: 0.86239, std: 0.00956, params: {'n_neighbors': 84},
mean: 0.86257, std: 0.01001, params: {'n_neighbors': 85},
mean: 0.86088, std: 0.01023, params: {'n_neighbors': 86},
mean: 0.86088, std: 0.01051, params: {'n_neighbors': 87},
mean: 0.85981, std: 0.01214, params: {'n_neighbors': 88},
mean: 0.85937, std: 0.01208, params: {'n_neighbors': 89},
mean: 0.85830, std: 0.01087, params: {'n_neighbors': 90},
mean: 0.85821, std: 0.01108, params: {'n_neighbors': 91},
mean: 0.85616, std: 0.00995, params: {'n_neighbors': 92},
mean: 0.85652, std: 0.00954, params: {'n_neighbors': 93},
mean: 0.85519, std: 0.00892, params: {'n_neighbors': 94},
mean: 0.85545, std: 0.00957, params: {'n_neighbors': 95},
mean: 0.85368, std: 0.00900, params: {'n_neighbors': 96},
mean: 0.85332, std: 0.00845, params: {'n_neighbors': 97},
mean: 0.85199, std: 0.00923, params: {'n_neighbors': 98},
mean: 0.85190, std: 0.00926, params: {'n_neighbors': 99},
mean: 0.85021, std: 0.00825, params: {'n_neighbors': 100},
mean: 0.85039, std: 0.00930, params: {'n_neighbors': 101},
mean: 0.84834, std: 0.01015, params: {'n_neighbors': 102},
mean: 0.84825, std: 0.01016, params: {'n_neighbors': 103},
mean: 0.84559, std: 0.00954, params: {'n_neighbors': 104},
mean: 0.84559, std: 0.00922, params: {'n_neighbors': 105},
mean: 0.84416, std: 0.00977, params: {'n_neighbors': 106},
mean: 0.84452, std: 0.00978, params: {'n_neighbors': 107},
mean: 0.84372, std: 0.00935, params: {'n_neighbors': 108},
mean: 0.84399, std: 0.00922, params: {'n_neighbors': 109},
mean: 0.84256, std: 0.00943, params: {'n_neighbors': 110},
mean: 0.84239, std: 0.00930, params: {'n_neighbors': 111},
mean: 0.84132, std: 0.00941, params: {'n_neighbors': 112},
mean: 0.84087, std: 0.00924, params: {'n_neighbors': 113},
mean: 0.84034, std: 0.00864, params: {'n_neighbors': 114},
mean: 0.84016, std: 0.00875, params: {'n_neighbors': 115},
mean: 0.83981, std: 0.00849, params: {'n_neighbors': 116},
mean: 0.83972, std: 0.00862, params: {'n_neighbors': 117},
mean: 0.83865, std: 0.00779, params: {'n_neighbors': 118},
mean: 0.83892, std: 0.00782, params: {'n_neighbors': 119},
mean: 0.83883, std: 0.00798, params: {'n_neighbors': 120},
mean: 0.83892, std: 0.00810, params: {'n_neighbors': 121},
mean: 0.83661, std: 0.00577, params: {'n_neighbors': 122},
mean: 0.83705, std: 0.00582, params: {'n_neighbors': 123},
mean: 0.83670, std: 0.00582, params: {'n_neighbors': 124},
mean: 0.83714, std: 0.00545, params: {'n_neighbors': 125},
mean: 0.83732, std: 0.00565, params: {'n_neighbors': 126},
mean: 0.83741, std: 0.00610, params: {'n_neighbors': 127},
mean: 0.83732, std: 0.00647, params: {'n_neighbors': 128},
mean: 0.83741, std: 0.00634, params: {'n_neighbors': 129},
mean: 0.83759, std: 0.00626, params: {'n_neighbors': 130},
mean: 0.83750, std: 0.00609, params: {'n_neighbors': 131},
mean: 0.83759, std: 0.00639, params: {'n_neighbors': 132},
mean: 0.83767, std: 0.00616, params: {'n_neighbors': 133},
mean: 0.83732, std: 0.00640, params: {'n_neighbors': 134},
mean: 0.83776, std: 0.00641, params: {'n_neighbors': 135},
mean: 0.83723, std: 0.00690, params: {'n_neighbors': 136},
mean: 0.83794, std: 0.00675, params: {'n_neighbors': 137},
mean: 0.83687, std: 0.00710, params: {'n_neighbors': 138},
mean: 0.83687, std: 0.00718, params: {'n_neighbors': 139},
mean: 0.83607, std: 0.00744, params: {'n_neighbors': 140},
mean: 0.83625, std: 0.00725, params: {'n_neighbors': 141},
mean: 0.83501, std: 0.00739, params: {'n_neighbors': 142},
mean: 0.83536, std: 0.00761, params: {'n_neighbors': 143},
mean: 0.83430, std: 0.00774, params: {'n_neighbors': 144},
mean: 0.83536, std: 0.00776, params: {'n_neighbors': 145},
mean: 0.83510, std: 0.00746, params: {'n_neighbors': 146},
mean: 0.83527, std: 0.00732, params: {'n_neighbors': 147},
mean: 0.83412, std: 0.00716, params: {'n_neighbors': 148},
mean: 0.83492, std: 0.00704, params: {'n_neighbors': 149},
mean: 0.83474, std: 0.00740, params: {'n_neighbors': 150},
mean: 0.83483, std: 0.00759, params: {'n_neighbors': 151},
mean: 0.83492, std: 0.00751, params: {'n_neighbors': 152},
mean: 0.83536, std: 0.00788, params: {'n_neighbors': 153},
mean: 0.83492, std: 0.00833, params: {'n_neighbors': 154},
mean: 0.83492, std: 0.00841, params: {'n_neighbors': 155},
mean: 0.83492, std: 0.00815, params: {'n_neighbors': 156},
mean: 0.83510, std: 0.00822, params: {'n_neighbors': 157},
mean: 0.83527, std: 0.00826, params: {'n_neighbors': 158},
mean: 0.83554, std: 0.00832, params: {'n_neighbors': 159},
mean: 0.83554, std: 0.00778, params: {'n_neighbors': 160},
mean: 0.83519, std: 0.00745, params: {'n_neighbors': 161},
mean: 0.83519, std: 0.00757, params: {'n_neighbors': 162},
mean: 0.83536, std: 0.00715, params: {'n_neighbors': 163},
mean: 0.83527, std: 0.00767, params: {'n_neighbors': 164},
mean: 0.83536, std: 0.00778, params: {'n_neighbors': 165},
mean: 0.83492, std: 0.00759, params: {'n_neighbors': 166},
mean: 0.83510, std: 0.00790, params: {'n_neighbors': 167},
mean: 0.83527, std: 0.00820, params: {'n_neighbors': 168},
mean: 0.83492, std: 0.00775, params: {'n_neighbors': 169},
mean: 0.83456, std: 0.00746, params: {'n_neighbors': 170},
mean: 0.83447, std: 0.00782, params: {'n_neighbors': 171},
mean: 0.83376, std: 0.00770, params: {'n_neighbors': 172},
mean: 0.83376, std: 0.00731, params: {'n_neighbors': 173},
mean: 0.83270, std: 0.00737, params: {'n_neighbors': 174},
mean: 0.83287, std: 0.00736, params: {'n_neighbors': 175},
mean: 0.83261, std: 0.00701, params: {'n_neighbors': 176},
mean: 0.83234, std: 0.00676, params: {'n_neighbors': 177},
mean: 0.83225, std: 0.00719, params: {'n_neighbors': 178},
mean: 0.83207, std: 0.00718, params: {'n_neighbors': 179},
mean: 0.83181, std: 0.00746, params: {'n_neighbors': 180},
mean: 0.83190, std: 0.00741, params: {'n_neighbors': 181},
mean: 0.83243, std: 0.00808, params: {'n_neighbors': 182},
mean: 0.83261, std: 0.00827, params: {'n_neighbors': 183},
mean: 0.83225, std: 0.00819, params: {'n_neighbors': 184},
mean: 0.83279, std: 0.00779, params: {'n_neighbors': 185},
mean: 0.83118, std: 0.00688, params: {'n_neighbors': 186},
mean: 0.83154, std: 0.00684, params: {'n_neighbors': 187},
mean: 0.82950, std: 0.00716, params: {'n_neighbors': 188},
mean: 0.83003, std: 0.00689, params: {'n_neighbors': 189},
mean: 0.82870, std: 0.00649, params: {'n_neighbors': 190},
mean: 0.82878, std: 0.00704, params: {'n_neighbors': 191},
mean: 0.82665, std: 0.00624, params: {'n_neighbors': 192},
mean: 0.82718, std: 0.00654, params: {'n_neighbors': 193},
mean: 0.82478, std: 0.00895, params: {'n_neighbors': 194},
mean: 0.82550, std: 0.00846, params: {'n_neighbors': 195},
mean: 0.82230, std: 0.00974, params: {'n_neighbors': 196},
mean: 0.82274, std: 0.01041, params: {'n_neighbors': 197},
mean: 0.81936, std: 0.00926, params: {'n_neighbors': 198},
mean: 0.81972, std: 0.00963, params: {'n_neighbors': 199}]
In [104]:
# create a list of the mean scores only
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print(grid_mean_scores)
[0.9623966574806649, 0.9588407858476309, 0.9456840608054049, 0.94621744155036, 0.9387501111209885, 0.9428393634989777, 0.9379500400035559, 0.9383945239576851, 0.9335052004622633, 0.9349275491154769, 0.9304827095741843, 0.9325273357631789, 0.9293270512934483, 0.9296826384567517, 0.9271935283136279, 0.9281713930127122, 0.9255933860787625, 0.9272824251044537, 0.9249711085429816, 0.9247044181705041, 0.9220375144457285, 0.9218597208640769, 0.9190150235576495, 0.9177704684860877, 0.9156369455062672, 0.914570184016357, 0.9124366610365365, 0.9124366610365365, 0.911014312383323, 0.9111032091741488, 0.9090585829851542, 0.9094141701484576, 0.9077251311227664, 0.9072806471686372, 0.9055027113521202, 0.9041692594897324, 0.9029247044181705, 0.9009689750200017, 0.9003466974842208, 0.8987465552493555, 0.8975908969686194, 0.8962574451062316, 0.8953684771979732, 0.8927015734731976, 0.8917237087741132, 0.8895012890034669, 0.8890568050493377, 0.8866565916970397, 0.8857676237887813, 0.8852342430438261, 0.8845230687172193, 0.8837229975997867, 0.8828340296915281, 0.8814116810383145, 0.881144990665837, 0.8806116099208818, 0.879900435594275, 0.8791003644768424, 0.8789225708951907, 0.8780336029869322, 0.8776780158236288, 0.8768779447061961, 0.8765223575428926, 0.8759000800071117, 0.8755444928438083, 0.8747444217263757, 0.8741221441905948, 0.873944350608943, 0.8734109698639879, 0.8731442794915104, 0.8726997955373811, 0.8722553115832519, 0.871544137256645, 0.8707440661392124, 0.8705662725575607, 0.8700328918126056, 0.8696773046493022, 0.8675437816694818, 0.8669215041337007, 0.8656769490621389, 0.8659436394346164, 0.8646101875722286, 0.8638990132456218, 0.8623877678015823, 0.862565561383234, 0.8608765223575429, 0.8608765223575429, 0.8598097608676327, 0.8593652769135034, 0.8582985154235933, 0.8582096186327673, 0.8561649924437728, 0.8565205796070762, 0.8551871277446884, 0.8554538181171659, 0.853675882300649, 0.8533202951373455, 0.8519868432749578, 0.851897946484132, 0.8502089074584408, 0.8503867010400925, 0.8483420748510979, 0.848253178060272, 0.8455862743354965, 0.8455862743354965, 0.8441639256822828, 0.8445195128455862, 0.8437194417281536, 0.8439861321006311, 0.8425637834474176, 0.8423859898657658, 0.8413192283758556, 0.8408747444217264, 0.8403413636767713, 0.8401635700951195, 0.8398079829318161, 0.8397190861409903, 0.8386523246510801, 0.8389190150235577, 0.8388301182327318, 0.8389190150235577, 0.8366076984620855, 0.8370521824162148, 0.8366965952529114, 0.8371410792070406, 0.8373188727886923, 0.8374077695795181, 0.8373188727886923, 0.8374077695795181, 0.8375855631611698, 0.837496666370344, 0.8375855631611698, 0.8376744599519957, 0.8373188727886923, 0.8377633567428215, 0.8372299759978665, 0.8379411503244732, 0.8368743888345631, 0.8368743888345631, 0.8360743177171304, 0.8362521112987821, 0.8350075562272202, 0.8353631433905236, 0.8342963819006134, 0.8353631433905236, 0.835096453018046, 0.8352742465996977, 0.8341185883189617, 0.8349186594363943, 0.8347408658547426, 0.8348297626455685, 0.8349186594363943, 0.8353631433905236, 0.8349186594363943, 0.8349186594363943, 0.8349186594363943, 0.835096453018046, 0.8352742465996977, 0.8355409369721754, 0.8355409369721754, 0.8351853498088719, 0.8351853498088719, 0.8353631433905236, 0.8352742465996977, 0.8353631433905236, 0.8349186594363943, 0.835096453018046, 0.8352742465996977, 0.8349186594363943, 0.8345630722730909, 0.8344741754822651, 0.8337630011556583, 0.8337630011556583, 0.832696239665748, 0.8328740332473997, 0.8326073428749222, 0.8323406525024447, 0.8322517557116188, 0.8320739621299671, 0.8318072717574896, 0.8318961685483154, 0.8324295492932705, 0.8326073428749222, 0.8322517557116188, 0.8327851364565739, 0.8311849942217086, 0.831540581385012, 0.8294959551960174, 0.8300293359409725, 0.8286958840785847, 0.8287847808694107, 0.8266512578895902, 0.8271846386345453, 0.8247844252822473, 0.8254955996088541, 0.8222953151391235, 0.8227397990932528, 0.8193617210418704, 0.8197173082051737]
In [105]:
# plot the results (Manhattan distance)
plt.plot(k_potentials, grid_mean_scores)
plt.xlabel('Potential K values for KNN')
plt.ylabel('Cross-Validated Accuracy')
Out[105]:
<matplotlib.text.Text at 0x120cb5a50>

In [106]:
# plot the results (using Euclidean distance)
plt.plot(k_potentials, grid_mean_scores)
plt.xlabel('Potential K values for KNN')
plt.ylabel('Cross-Validated Accuracy')
Out[106]:
<matplotlib.text.Text at 0x12143b7d0>

In [107]:
# examine the best model
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
0.962396657481
{'n_neighbors': 1}
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
metric_params=None, n_jobs=1, n_neighbors=1, p=2,
weights='uniform')
In [ ]:

In [108]:
# read in the data & create matrices
X = X_scaled[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", \
"time_spend_company", "Work_accident", "promotion_last_5years", "low", "medium",\
"technical", "support", "IT", "product_mng", "marketing", "RandD", "accounting",\
"hr"]]
y = X_scaled["left"]
In [176]:
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
Support Vector Machines
In [177]:
# Train, fit and predict with SVM

from sklearn import svm
Model=svm.SVC(kernel='linear')
Model.fit(X_train,y_train)
Y_pred=Model.predict(X_test)
In [178]:
# Cross validation score for SVM
scores = cross_val_score(Model, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.780694712068
In [179]:
# Evaluate SVM for accuracy

from sklearn import metrics
metrics.accuracy_score(y_test,Y_pred)
Out[179]:
0.78106666666666669
In [180]:
# Confusion matrix for SVM

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2696  166]
[ 655  233]]
In [181]:
# Classification Accuracy confirmation between manual calculation and formula for SVM

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.781066666667
0.781066666667
In [182]:
# Misclassification Accuracy confirmation between manual calculation and formula for SVM
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.218933333333
0.218933333333
In [183]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated SVM

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.262387387387
0.262387387387
In [184]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for SVM

print(TN / float(TN + FP))
0.941998602376
In [185]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for SVM

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.583959899749
0.583959899749
In [186]:
# Misclassification Accuracy confirmation between manual calculation and formula for SVM
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.218933333333
0.218933333333
In [187]:
# Roc_curve for Support Vector Machine Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [92]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [93]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Support Vector Machines: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

Random Forest
In [157]:
# Random Forests Classifier from Scikit Learn

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_jobs=2)
RF.fit(X_train,y_train)
Y_pred=RF.predict(X_test)
In [158]:
# Score for 10-fold Cross Validation for Random Forests Classifier

scores = cross_val_score(RF, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.986309602501
In [159]:
# Prediction Accuracy for Random Forests

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.9896
In [160]:
# Confusion matrix for Random Forests

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2857    5]
[  34  854]]
In [161]:
# Classification Accuracy confirmation between manual calculation and formula for Random Forest

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.9896
0.9896
In [162]:
# Misclassification Accuracy confirmation between manual calculation and formula for Random Forest
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0104
0.0104
In [163]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for Random Forest Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.961711711712
0.961711711712
In [164]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for Random Forest Classifier

print(TN / float(TN + FP))
0.998252969951
In [165]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for Random Forest Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.994179278231
0.994179278231
In [166]:
# Misclassification Accuracy confirmation between manual calculation and formula for Random Forest
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0104
0.0104
In [167]:
# Roc_curve for Random Forest Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [168]:
y_pred_prob_Random_Forest = RF.predict_proba(X_test)[:, 1]
In [172]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_Random_Forest, bins=5)
plt.xlim(0, 1)
plt.title('Random Forest Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[172]:
<matplotlib.text.Text at 0x12013a8d0>

In [170]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [171]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Random Forest: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

Bagging
In [109]:
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
In [110]:
# Bagging Classifier

seed = 7
cart = DecisionTreeClassifier()
num_trees = 100
Bagging = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
Bagging.fit(X_train,y_train)
Y_pred=Bagging.predict(X_test)
In [111]:
# Cross Validation for Bagging

scores = cross_val_score(Bagging, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.987910076856
In [112]:
# Prediction Accuracy for Bagging

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.989333333333
In [113]:
# Confusion matrix for Bagging

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2854    8]
[  32  856]]
In [114]:
# Classification Accuracy confirmation between manual calculation and formula for Bagging

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.989333333333
0.989333333333
In [115]:
# Misclassification Accuracy confirmation between manual calculation and formula for Bagging
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0106666666667
0.0106666666667
In [116]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for Bagging Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.963963963964
0.963963963964
In [117]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for Bagging Classifier

print(TN / float(TN + FP))
0.997204751922
In [118]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for Bagging Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.990740740741
0.990740740741
In [119]:
# Misclassification Accuracy confirmation between manual calculation and formula for Bagging
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0106666666667
0.0106666666667
In [120]:
# Roc_curve for Bagging


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [121]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
In [122]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Bagging: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [123]:
y_pred_prob_Bagging = Bagging.predict_proba(X_test)[:, 1]
In [124]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_Bagging, bins=5)
plt.xlim(0, 1)
plt.title('Bagging Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[124]:
<matplotlib.text.Text at 0x11b217610>

In [ ]:

AdaBoost
In [125]:
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
In [126]:
# AdaBoost Classifier

seed = 7
cart = AdaBoostClassifier()
num_trees = 100
AdaBoostClassifier = AdaBoostClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
AdaBoostClassifier.fit(X_train,y_train)
Y_pred=AdaBoostClassifier.predict(X_test)
In [127]:
# Cross Validation for AdaBoost

scores = cross_val_score(AdaBoostClassifier, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.958576016468
In [128]:
# Prediction Accuracy for AdaBoost

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.958133333333
In [129]:
# Confusion matrix for AdaBoost

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2793   69]
[  88  800]]
In [130]:
# Classification Accuracy confirmation between manual calculation and formula for Bagging

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.958133333333
0.958133333333
In [131]:
# Misclassification Accuracy confirmation between manual calculation and formula for AdaBoost
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0418666666667
0.0418666666667
In [132]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for AdaBoost Boosting Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.900900900901
0.900900900901
In [133]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for AdaBoost Classifier

print(TN / float(TN + FP))
0.975890985325
In [134]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for AdaBoost Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.920598388953
0.920598388953
In [135]:
# Misclassification Accuracy confirmation between manual calculation and formula for AdaBoost
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0418666666667
0.0418666666667
In [136]:
# Roc_curve for AdaBoost


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [137]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
In [138]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='AdaBoost: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [139]:
y_pred_prob_AdaBoost = AdaBoostClassifier.predict_proba(X_test)[:, 1]
In [140]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_AdaBoost, bins=5)
plt.xlim(0, 1)
plt.title('AdaBoost Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[140]:
<matplotlib.text.Text at 0x11b367a10>

In [ ]:

In [ ]:

In [ ]:

Gradient Boosting
In [141]:
from sklearn.ensemble import GradientBoostingClassifier
In [142]:
# Train and predict Gradient Boosting

GradientBoosting = GradientBoostingClassifier()
GradientBoosting.fit(X_train,y_train)
Y_pred=GradientBoosting.predict(X_test)
In [143]:
# Cross Validation for Gradient Boosting

scores = cross_val_score(GradientBoosting, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.976443246546
In [144]:
# Prediction Accuracy for Gradient Boosting

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.974133333333
In [145]:
# Confusion matrix for Gradient Boosting

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2832   30]
[  67  821]]
In [146]:
# Classification Accuracy confirmation between manual calculation and formula for Gradient Boosting

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.974133333333
0.974133333333
In [147]:
# Misclassification Accuracy confirmation between manual calculation and formula for Gradient Boosting
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0258666666667
0.0258666666667
In [148]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for Gradient Boosting Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.92454954955
0.92454954955
In [149]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for Gradient Classifier

print(TN / float(TN + FP))
0.989517819706
In [150]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for Gradient Boosting Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.964747356052
0.964747356052
In [151]:
# Misclassification Accuracy confirmation between manual calculation and formula for Gradient Boosting
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0258666666667
0.0258666666667
In [152]:
# Roc_curve for Gradient Boosting


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Gradient Boosting Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [153]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
In [154]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Gradient Boosting: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [155]:
y_pred_prob_GradientBoosting = GradientBoosting.predict_proba(X_test)[:, 1]
In [156]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_GradientBoosting, bins=5)
plt.xlim(0, 1)
plt.title('Gradient Boosting Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[156]:
<matplotlib.text.Text at 0x11c2566d0>

In [ ]:

In [ ]:

Part IV
Conclusion
One interesting chart showed us which departments have the most people leaving (and staying). They are Sales followed by Technical and Support. That means that the biggest effort at the company should be placed on keeping the strong employees in those departments (but by no means neglect the other departments).
There are several other interesting patterns in the data. Some are very intuitive whereas others very surprising and unexpected. There are three clusters of people leaving the company (the most important cluster is not as dense, in the sense that there are actually slightly more people there that stay than leave, but nevertheless its the most interesting and surprising cluster).
The three groups are employees with high performance and low satisfaction, employees with low satisfcation and performance (rather obvious) and employees with high satisfaction and performance (a surprising result). The first two results are very strong but rather trivial, there is limited use in providing detailed analyses of them. However, the somewhat less dense but very significant group of highly satisfied employees that do remarkably well in their evaluations is of extreme importance. Why do they leave? Well, there are several important factors to consider.
The people with high salaries and at even average satisfaction levels rarely leave the firm. Hence, one attribute that people who tend to leave will generally possess is salaries that are not high and fall in the low or medium range. It is interesting to note though that people with near perfect to perfect satisfaction levels do not leave even if they are in the low or medium income range brackets. Hence, a possible course of action to potentially alleviate this issue is increase salaries for good employees. Also, a further investigation into what might be the differentiating factor between people who are very satisfied (0.7-0.9 rate) vs super satisfied (0.9-1.0) can be done to see if it is possible to further increase the satisfaction level of those that are very satisfied but not perfectly so (maybe its possible to do this while at the same time saving money by not instituting as much pay increases if there is a way to get them satisfied in another manner, perhaps by working less hours for instance) would be a good idea.
Secondly, promotions are a huge factor. Looking at the plot of promotions during the past five years, it is very clear that people who are promoted stay and people who are not promoted leave. Of course, promotions and salary increases have some correlation because people tend to receive salary increases during promotions. Hence, it's important to promote employees if they do good work! That will help decrease the turnover rate.
The third important point is that employees who work a lot of hours that stay at the company for at least a medium amount of time tend to leave. Only the relatively small number of employees who work at the company for many years stay at the company regardless of working hours but very few make it there. Hence, the hours worked plays an important decision making role when employees think about whether to stay at the firm or seek other employment. Due to this finding it is important to see if any kind of rebalancing can be done so that part of the work can perhaps be shifted between employees who do not work as many hours and those that do in order to make the distribution of hours worked by employees more uniform which can perhaps increase employees willingness to stay at the company longer. It is quiet possible that some employees just feel burnt out even if they are generally happy with their job. Due to an inefficient work life balance, they might seek to find employment elsewhere.
Multiple machine learning algorithms were ran on the data and quiet a few did relatively well in terms of making predictions on out of sample data as measured by the accuracy score, the confusion matrix and deriatives thereof as well as the ROC Curve. In particular, the Random Forest and Bagging algorithms did the best with accuracy and cross-validation scores marginally different from each other just under 99%. Both correctly predicted over 3,700 out of 3,750 employees accurately (compared to a baseline accuracy of only 76% (2,850) as to whether they would leave/stay). The various scores computed from the confusion matrix were all almost perfect. Particularly importantly, the ROC curve has an area covering of .98 for both those algorithms.
Hence, using either of these two algorithms (Random Forests or Bagging) have given us great predictions as to which employees will leave vs which ones would stay.
Assumptions
We have assumed the dataset is accurate as we have not audited its validity.
Another assumption we made is that we trained on 75% of the data, conducted cross-validation, and then tested on the 25% remaining out of sample. However, when trying to sample to train a larger sample of the data (such as 90/10 test/train split), we received very similar results in our algorithms. Hence, our model is robust as should perform well when used on future data.
An assumption made by the Random Forest Model which did exceptionally well is that the sampling that it conducted during boostrap aggregation is actually representative. This is a safe assumption to make and the model has done really well on out of sample data.
Introduction:
This project analyzes a dataset in an attempt to determine what causes the good employees to leave their current jobs. The first step is to clean the data (Part I) and visualize it (Part II) using various types of graphs and charts to help establish clear relationships between various features. Part III analyzes the dataset using various Machine Learning algorithms and picks the best algorithm to model this dataset based on various metrics. Lastly, a conclusion on the entire analysis is provided.
Part I
Data Cleaning
This initial stage of the project is to clean the data in such a way that all the important attributes of the dataset be retained and unhindered for analysis purposes. Hence, the initial cleaning phase of the project imports the dataset and the libraries needed for the cleaning stages of the project. A check for Null Values/Missing Values is conducted (none are present). The multiple categorical variables in the dataset are created into dummy variables which is necessary for later stages of the analysis. These new variables are afterwards added to the original dataset and the initial variables from which dummification was done are removed to avoid redundancy. Lastly, the dataset is normalized due to the need to apply machine learning algorithms to it in later stages of the project.
In [1]:
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

%matplotlib inline
In [2]:
HR = pd.read_csv('HR_comma_sep.csv')
HR.head(10)
Out[2]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary
0	0.38	0.53	2	157	3	0	1	0	sales	low
1	0.80	0.86	5	262	6	0	1	0	sales	medium
2	0.11	0.88	7	272	4	0	1	0	sales	medium
3	0.72	0.87	5	223	5	0	1	0	sales	low
4	0.37	0.52	2	159	3	0	1	0	sales	low
5	0.41	0.50	2	153	3	0	1	0	sales	low
6	0.10	0.77	6	247	4	0	1	0	sales	low
7	0.92	0.85	5	259	5	0	1	0	sales	low
8	0.89	1.00	5	224	5	0	1	0	sales	low
9	0.42	0.53	2	142	3	0	1	0	sales	low
In [3]:
HR[HR['satisfaction_level'] == np.nan]
HR[HR['last_evaluation'] == np.nan]
HR[HR['number_project'] == np.nan]
HR[HR['average_montly_hours'] == np.nan]
HR[HR['time_spend_company'] == np.nan]
HR[HR['Work_accident'] == np.nan]
HR[HR['left'] == np.nan]
HR[HR['sales'] == np.nan]
HR[HR['salary'] == np.nan]
Out[3]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary
In [4]:
# Create a heatmap, to check for null values.

plt.figure(figsize=(10,7))
sns.heatmap(HR.isnull(),yticklabels=False, cbar=False, cmap='viridis')
Out[4]:
<matplotlib.axes._subplots.AxesSubplot at 0x114c6ee50>

In [5]:
HR['sales'].value_counts()
Out[5]:
sales          4140
technical      2720
support        2229
IT             1227
product_mng     902
marketing       858
RandD           787
accounting      767
hr              739
management      630
Name: sales, dtype: int64
In [6]:
HR['salary'].value_counts()
Out[6]:
low       7316
medium    6446
high      1237
Name: salary, dtype: int64
In [7]:
HR['left'].value_counts()
Out[7]:
0    11428
1     3571
Name: left, dtype: int64
In [8]:
# Create dummy variables for all the departments

sales = pd.get_dummies(HR.sales)

sales.columns = ["IT", "RandD", "accounting", "hr", "management", "marketing", "product_mng", "sales_department", "support", "technical"]

sales.head()
Out[8]:
IT	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical
0	0	0	0	0	0	0	0	1	0	0
1	0	0	0	0	0	0	0	1	0	0
2	0	0	0	0	0	0	0	1	0	0
3	0	0	0	0	0	0	0	1	0	0
4	0	0	0	0	0	0	0	1	0	0
In [9]:
# Create dummy variables for the salary

salary = pd.get_dummies(HR.salary)

salary.columns = ['low', 'medium', 'high']

salary.head()
Out[9]:
low	medium	high
0	0	1	0
1	0	0	1
2	0	0	1
3	0	1	0
4	0	1	0
In [10]:
# Create dummy variable for stayed/left

left = pd.get_dummies(HR.left)

left.columns = ['left', 'stayed']

left.head(10)
Out[10]:
left	stayed
0	0	1
1	0	1
2	0	1
3	0	1
4	0	1
5	0	1
6	0	1
7	0	1
8	0	1
9	0	1
In [11]:
# Concatenate the original dataset with the newly created salary and sales dummified variables

HR = pd.concat([HR, salary, sales], axis = 1)

HR.head(10)
Out[11]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary	...	IT	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical
0	0.38	0.53	2	157	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
1	0.80	0.86	5	262	6	0	1	0	sales	medium	...	0	0	0	0	0	0	0	1	0	0
2	0.11	0.88	7	272	4	0	1	0	sales	medium	...	0	0	0	0	0	0	0	1	0	0
3	0.72	0.87	5	223	5	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
4	0.37	0.52	2	159	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
5	0.41	0.50	2	153	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
6	0.10	0.77	6	247	4	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
7	0.92	0.85	5	259	5	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
8	0.89	1.00	5	224	5	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
9	0.42	0.53	2	142	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
10 rows × 23 columns
In [12]:
HR.dtypes
Out[12]:
satisfaction_level       float64
last_evaluation          float64
number_project             int64
average_montly_hours       int64
time_spend_company         int64
Work_accident              int64
left                       int64
promotion_last_5years      int64
sales                     object
salary                    object
low                        uint8
medium                     uint8
high                       uint8
IT                         uint8
RandD                      uint8
accounting                 uint8
hr                         uint8
management                 uint8
marketing                  uint8
product_mng                uint8
sales_department           uint8
support                    uint8
technical                  uint8
dtype: object
In [13]:
# Create a new column 

salary_map = {'low': 1, 'medium': 2, 'high': 3}

HR['salary_variable'] = HR['salary'].apply(lambda x: salary_map[x])

HR.head()
Out[13]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary	...	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical	salary_variable
0	0.38	0.53	2	157	3	0	1	0	sales	low	...	0	0	0	0	0	0	1	0	0	1
1	0.80	0.86	5	262	6	0	1	0	sales	medium	...	0	0	0	0	0	0	1	0	0	2
2	0.11	0.88	7	272	4	0	1	0	sales	medium	...	0	0	0	0	0	0	1	0	0	2
3	0.72	0.87	5	223	5	0	1	0	sales	low	...	0	0	0	0	0	0	1	0	0	1
4	0.37	0.52	2	159	3	0	1	0	sales	low	...	0	0	0	0	0	0	1	0	0	1
5 rows × 24 columns
In [14]:
del HR['salary']

del HR['sales']
In [15]:
HR.head()
Out[15]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	low	medium	...	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical	salary_variable
0	0.38	0.53	2	157	3	0	1	0	0	1	...	0	0	0	0	0	0	1	0	0	1
1	0.80	0.86	5	262	6	0	1	0	0	0	...	0	0	0	0	0	0	1	0	0	2
2	0.11	0.88	7	272	4	0	1	0	0	0	...	0	0	0	0	0	0	1	0	0	2
3	0.72	0.87	5	223	5	0	1	0	0	1	...	0	0	0	0	0	0	1	0	0	1
4	0.37	0.52	2	159	3	0	1	0	0	1	...	0	0	0	0	0	0	1	0	0	1
5 rows × 22 columns
In [16]:
HR.dtypes
Out[16]:
satisfaction_level       float64
last_evaluation          float64
number_project             int64
average_montly_hours       int64
time_spend_company         int64
Work_accident              int64
left                       int64
promotion_last_5years      int64
low                        uint8
medium                     uint8
high                       uint8
IT                         uint8
RandD                      uint8
accounting                 uint8
hr                         uint8
management                 uint8
marketing                  uint8
product_mng                uint8
sales_department           uint8
support                    uint8
technical                  uint8
salary_variable            int64
dtype: object
In [17]:
from sklearn.preprocessing import MinMaxScaler
X_scaled = MinMaxScaler().fit_transform(HR.astype(float))
X_scaled = pd.DataFrame(X_scaled, columns = HR.columns)

# Min Max Scaler

X_scaled.head(10)
Out[17]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	low	medium	...	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical	salary_variable
0	0.318681	0.265625	0.0	0.285047	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
1	0.780220	0.781250	0.6	0.775701	0.500	0.0	1.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.5
2	0.021978	0.812500	1.0	0.822430	0.250	0.0	1.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.5
3	0.692308	0.796875	0.6	0.593458	0.375	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
4	0.307692	0.250000	0.0	0.294393	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
5	0.351648	0.218750	0.0	0.266355	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
6	0.010989	0.640625	0.8	0.705607	0.250	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
7	0.912088	0.765625	0.6	0.761682	0.375	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
8	0.879121	1.000000	0.6	0.598131	0.375	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
9	0.362637	0.265625	0.0	0.214953	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
10 rows × 22 columns
Part II
Exploratory Data Analysis (EDA)
In this section, we try to visualize the data on employees using various plots, graphs, charts in order to get an idea of how different features in the dataset relate to each other.
The below represents a heatmap of correlations of features. Some interesting observations from this heatmap: Quiet intuitively, you can see that the relationship between satisfaction level and people that left the company is negative. It's also interesting to note that there is a positive correlation between the number of hours employees work and their evaluation. These and other relationships will be explored more deeply in the rest of the EDA.
In [18]:
correlation = X_scaled.corr()
plt.figure(figsize=(20,20))
sns.heatmap(correlation, vmax=10, square=True,annot=True,cmap=None, linewidths=1)

plt.title('Correlation between features')
Out[18]:
<matplotlib.text.Text at 0x118f3a050>

In [19]:
# Exported csv file to work with in Tableau
X_scaled.to_csv("Updated_Capstone.csv")
In [20]:
X_scaled['left'].value_counts()
Out[20]:
0.0    11428
1.0     3571
Name: left, dtype: int64
In [21]:
stayed = np.where(X_scaled['left'] == 0)[0]
left = np.where(X_scaled['left'] == 1)[0]
For a given level of satisfaction, there is definitely an intuitive pattern that the higher someone's salary is, the less they are likely to leave. This relationship is a lot more apparent in the distinction between the high income earners vs low & median income earners as a group. The difference in the people that left between low and median income salaries is not significant. Hence, most of the people that leave are not receiving a high salary, which means that you if you would like to retain these employees, you should increase their pay!
In [22]:
# plot of Satisfaction vs Last Evaluation
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.satisfaction_level[stayed], X_scaled.salary_variable[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.satisfaction_level[left], X_scaled.salary_variable[left], alpha=0.1, color='r')
ax.set_title("satisfaction_level vs salary")
ax.set_xlabel("satisfaction_level")
ax.set_ylabel("salary")
plt.show()

In the Satisfaction vs Last Evaluation plot below, we can note some interesting observations. Firstly, there is a sizeable cluster of people with strong evaluations and low satisfaction levels leaving. Secondly, there is a significant group of people leaving that are slighly below average in performance and evaluation. Lastly, and by far the most interesting observation of the three, there is a decent size of a not very dense cluster of people leaving that have done extremely well on their evaluations (0.7 to 1 on a 0-1 scale) and and have very high satisfaction rates, albeit not perfect (0.7-0.9 on a 0-1 scale). That particular group of employees also have many people that stay, but a significant proportion of that group is leaving. So the most surprising thing to observe is that people can be satisfied with the job and still leave! There are several possibilites for why this might be possible. People might enjoy their job but think they can do even better elsewhere, whether its income or career growth (or a combination of the two, considering that those things tend to go hand in hand). Another possibility is that people might not be honest in surveys because of fear of being reprimanded for negative surveys.
In [23]:
# plot of Satisfaction vs Last Evaluation
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.satisfaction_level[stayed], X_scaled.last_evaluation[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.satisfaction_level[left], X_scaled.last_evaluation[left], alpha=0.1, color='r')
ax.set_title("satisfaction_level vs last_evaluation")
ax.set_xlabel("satisfaction_level")
ax.set_ylabel("last_evaluation")
plt.show()

The Satisfaction vs Promotion during the last five years plot below shows a very strong relationship in several interesting ways! People who have been promoted don't leave very often! However, such promotions are not common enough! Therefore, if you don't want good employees to leave, promoting them will really help!
In [24]:
# plot of Satisfaction vs Promotion during last five years
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.satisfaction_level[stayed], X_scaled.promotion_last_5years[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.satisfaction_level[left], X_scaled.promotion_last_5years[left], alpha=0.1, color='r')
ax.set_title("satisfaction_level vs promotion_last_5years")
ax.set_xlabel("satisfaction_level")
ax.set_ylabel("promotion_last_5years")
plt.show()

The below scatter plot compares the average working hours against the time spent at the company and how it impacts the employee turnover rate. There are multiple interesting relationships to note. For employees that just started out, the majority tend to stick around for some time even if they work substantial hours. However, for employees who have been at the company for a while, there is a significant tendency to leave the company if working substantial hours. Consequently, we can observe that very few employees stick around at the company for a long time period because the ones who were working long hours almost all left! However, those that do stick around for a long time (relatively few) tend to all be devoted and not leave
There are several conjectures for why this is possible. One is that employees have worked for a significant amount of hours and put in a lot of time and effort expecting a quick promotion and/or pay increase (refer to graph above for relationship between promotions and staying/leaving) that they have not received and leave for elsewhere. Another reason might be that people have tried working for a little bit of time, realized that they are being overworked and have started looking for other opportunities to seek employment.
In [25]:
# plot of Average_monthly_hours vs time_spend_company
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.average_montly_hours[stayed], X_scaled.time_spend_company[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.average_montly_hours[left], X_scaled.time_spend_company[left], alpha=0.1, color='r')
ax.set_title("average_monthly_hours vs time_spend_company")
ax.set_xlabel("average_monthly_hours")
ax.set_ylabel("time_spend_company")
plt.show()

In [26]:
# Imported from Tableau
from IPython.display import Image
Image(filename = 'Average Monthly Hours vs Satisfaction Level.png', width = 1000)
Out[26]:

There does not appear to be a strong relationship between accidents and people leaving the company, hence, people stay/leave regardless of the accidents they might have experienced.
In [27]:
# plot of work_accident vs time_spend_company
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.Work_accident[stayed], X_scaled.time_spend_company[stayed], alpha=0.5, color='g')
ax.scatter(X_scaled.Work_accident[left], X_scaled.time_spend_company[left], alpha=0.5, color='r')
ax.set_title("Work_accident vs time_spend_company")
ax.set_xlabel("Work_accident")
ax.set_ylabel("time_spend_company")
plt.show()

The below stripplot compares number of projects done by an employee versus the number of promotions they have received over the past 5 years. It is interesting to note that the number of projects completed has very insignificant impact on whether a person leaves. On the contrary, the promotions received during the past 5 years is of critical importance. The majority of people who have received a promotion stay. Almost all those who were not promoted leave.
The data seems insufficient to be able to determine why the number of projects is not related to promotions. There are at least several possibilities. Perhaps the number of projects refers to the number assigned (vs completed) or that the projects are not all of equal difficulty and not equally time consuming leading to distorted hypothesis that the number of projects might necessarily correspond to an employee's productivity.
In [28]:
import seaborn as sns

sns.stripplot("number_project", "promotion_last_5years", data=X_scaled, hue="left", jitter = True)
Out[28]:
<matplotlib.axes._subplots.AxesSubplot at 0x117949d90>

The Satisfaction level distribution below shows that the majority of people are at least moderately satisfied with their job. Therefore, for the people that are satisfied and do well, efforts should be placed on making them even more happy by promoting them. For the people that do poorly or mediocre, either studies should be placed on determining why they are not doing well and trying to improve their performance or attempting to replace them with better employees (this would depend on a cost/benefit analysis) done by the company regarding this issue.
In [29]:
# Histogram distribution of Satisfaction Level

f,ax1 = plt.subplots(1,1)
sns.distplot(X_scaled['satisfaction_level'],bins=10,norm_hist=False)
plt.show()

The distribution (which is a decent approximation of a normal distribution with almost no tails and some kurtosis present) of the last evaluation graph shows us there is a sizeable portion of people that do well in their evaluations, hence the company should try to do as much as possible to keep those people!
In [30]:
# Histogram distribution of Last Evaluation


f,ax1 = plt.subplots(1,1)
sns.distplot(X_scaled['last_evaluation'],bins=10,norm_hist=False)
plt.show()

The below plot shows the ratios of people who stay in the company by department and the ratio of people who leave the company by department. It is clear from the chart that the departments with the most number of employees also have the highest ratios of both people who have left and those that stayed at the company. This tells us there should be particular effort place on retaining good employees that work in those departments, in particular the sales, technical and support departments, respectively. The largest efforts should be placed on retaining the good employees in these groups because they have the most people leaving.
In [34]:
# This plot is to compare the people that left across different departments
# The non-department variables are deleted to only keep the departments

del X_scaled['satisfaction_level']
del X_scaled['last_evaluation']
del X_scaled['number_project']
del X_scaled['average_montly_hours']
del X_scaled['time_spend_company']
del X_scaled['Work_accident']
del X_scaled['promotion_last_5years']
del X_scaled['low']
del X_scaled['medium']
del X_scaled['high']
del X_scaled['salary_variable']


left=X_scaled.groupby('left').mean()
left=left.transpose()
plt.figure(figsize=(20,10))
left.plot(kind='bar')
Out[34]:
<matplotlib.axes._subplots.AxesSubplot at 0x11a7bcd10>
<matplotlib.figure.Figure at 0x11a360850>

Part III
Predictive Modelling
In this section below, multiple models will be run that are applicable to classification. Afterwards, various scoring metrics will be calculated for the models and a determination will be made as to which model is based for this dataset.
In [174]:
# read in the data & create matrices
X = X_scaled[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", \
"time_spend_company", "Work_accident", "promotion_last_5years", "salary_variable",\
"technical", "support", "IT", "product_mng", "marketing", "RandD", "accounting",\
"hr", "management", "sales_department"]]
y = X_scaled["left"]
In [175]:
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
Logistic Regression Model
In [33]:
# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
model = LogisticRegression()

# fit the model with data
mdl = model.fit(X_train, y_train)

# predict the response for new observations

logit = model.predict(X_test)

# sklearn output to check that ran on right data split.

len(logit)
Out[33]:
3750
In [34]:
# These represent the y-intercept and coefficients for all the variables in the logistic regression.

print(model.intercept_)
print(model.coef_)
[ 0.57305398]
[[-3.71487625  0.45484967 -1.50619896  0.8704181   1.97590531 -1.45425383
-1.26024249 -1.32512177  0.22588571  0.23497924 -0.01282827  0.07509077
0.13925062 -0.47542127  0.26606552  0.3920348  -0.44451555  0.17251241]]
In [35]:
# Prediction Accuracy for Logistic Regression

from sklearn import metrics
print(metrics.accuracy_score(y_test, logit))
0.796
In [36]:
from sklearn.cross_validation import cross_val_score
In [37]:
# 10-fold cross-validation for Logistic Regression to find the CV score
mdl = model.fit(X_train, y_train)
scores = cross_val_score(mdl, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.794205514996
In [38]:
from sklearn import metrics
confusion = metrics.confusion_matrix(y_test, logit)
print ((confusion))
[[2651  211]
[ 554  334]]
In [39]:
y_pred_prob = model.predict_proba(X_test)[:, 1]
In [40]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob, bins=5)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[40]:
<matplotlib.text.Text at 0x1181a6a90>

In [41]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [42]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, logit)


plot_confusion_matrix(cm, title='LogisticRegression: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [43]:
# Roc_curve for Logistic Regression Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, logit)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Logistic Regression Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

K Nearest Nearbours Model
K=1
In [44]:
from sklearn.neighbors import KNeighborsClassifier
In [45]:
# K-Nearest Neighbours with K=1
knn = KNeighborsClassifier(n_neighbors=1, metric = 'euclidean')
In [46]:
y = knn.fit(X_train, y_train)

KNN = knn.predict(X_test)

# sklearn output to check that ran on right data split.

len(KNN)
Out[46]:
3750
In [47]:
# Cross fold validation for K=1

md2 = knn.fit(X_train, y_train)
scores = cross_val_score(md2, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.962043001151
In [48]:
# Prediction Accuracy.

from sklearn import metrics
print(metrics.accuracy_score(y_test, KNN))
0.967466666667
In [49]:
confusion = metrics.confusion_matrix(y_test, KNN)
print confusion
[[2775   87]
[  35  853]]
In [50]:
# Classification Accuracy confirmation between manual calculation and formula for K=1

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, KNN))
0.967466666667
0.967466666667
In [51]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=1
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN))
0.0325333333333
0.0325333333333
In [52]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for K=1

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, KNN))
0.960585585586
0.960585585586
In [53]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for K=1


print(TN / float(TN + FP))
0.969601677149
In [54]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for K=1

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, KNN))
0.907446808511
0.907446808511
In [55]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=1
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN))
0.0325333333333
0.0325333333333
In [56]:
y_pred_prob_knn = knn.predict_proba(X_test)[:, 1]
In [57]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_knn, bins=5)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[57]:
<matplotlib.text.Text at 0x1181d6b90>

In [58]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [59]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, KNN)


plot_confusion_matrix(cm, title='KNN for K=1: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [60]:
# Roc_curve for KNN=1 Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, KNN)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('KNN=1 Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

K=5
In [61]:
# K-Nearest Neighbours with K=5

knn = KNeighborsClassifier(n_neighbors=5, metric = 'euclidean')
y = knn.fit(X_train, y_train)
KNN5 = knn.predict(X_test)

# sklearn output to check that ran on right data split.

len(KNN5)
Out[61]:
3750
In [62]:
# Cross Validation for K=5 Cross-Fold Validation

md2 = knn.fit(X_train, y_train)
scores = cross_val_score(md2, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.938309179837
In [63]:
# Prediction Accuracy on KNN=5 for accuracy score

from sklearn import metrics
print(metrics.accuracy_score(y_test, KNN5))
0.941333333333
In [64]:
confusion = metrics.confusion_matrix(y_test, KNN5)
print confusion
[[2731  131]
[  89  799]]
In [65]:
# Classification Accuracy confirmation between manual calculation and formula for K=5

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, KNN5))
0.941333333333
0.941333333333
In [66]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=5
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN5))
0.0586666666667
0.0586666666667
In [67]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for K=5

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, KNN5))
0.899774774775
0.899774774775
In [68]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for K=5

print(TN / float(TN + FP))
0.954227812718
In [69]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for K=5

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, KNN5))
0.859139784946
0.859139784946
In [70]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=5
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN5))
0.0586666666667
0.0586666666667
In [71]:
# Roc_curve for KNN=5 Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, KNN5)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [72]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [73]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, KNN5)


plot_confusion_matrix(cm, title='KNN for N=5: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [74]:
y_pred_prob_knn = knn.predict_proba(X_test)[:, 1]
In [75]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_knn, bins=5)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[75]:
<matplotlib.text.Text at 0x11acdea10>

In [ ]:

Grid Search for optimal N value in KNN
In [101]:
from sklearn.grid_search import GridSearchCV
/Users/Misha/anaconda/lib/python2.7/site-packages/sklearn/grid_search.py:43: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.
DeprecationWarning)
In [102]:
k_potentials = list(range(1, 200))
print(k_potentials)
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]
In [103]:
# Set Nearest Neighbours equal to k
knn = KNeighborsClassifier(n_neighbors=k_potentials, metric = 'euclidean')
# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_potentials)
# instantiate the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
# fit the grid with data
grid.fit(X_train, y_train)
# view the results
grid.grid_scores_
Out[103]:
[mean: 0.96240, std: 0.00591, params: {'n_neighbors': 1},
mean: 0.95884, std: 0.00543, params: {'n_neighbors': 2},
mean: 0.94568, std: 0.00463, params: {'n_neighbors': 3},
mean: 0.94622, std: 0.00555, params: {'n_neighbors': 4},
mean: 0.93875, std: 0.00620, params: {'n_neighbors': 5},
mean: 0.94284, std: 0.00673, params: {'n_neighbors': 6},
mean: 0.93795, std: 0.00746, params: {'n_neighbors': 7},
mean: 0.93839, std: 0.00819, params: {'n_neighbors': 8},
mean: 0.93351, std: 0.00932, params: {'n_neighbors': 9},
mean: 0.93493, std: 0.00848, params: {'n_neighbors': 10},
mean: 0.93048, std: 0.00825, params: {'n_neighbors': 11},
mean: 0.93253, std: 0.00810, params: {'n_neighbors': 12},
mean: 0.92933, std: 0.00898, params: {'n_neighbors': 13},
mean: 0.92968, std: 0.00824, params: {'n_neighbors': 14},
mean: 0.92719, std: 0.00747, params: {'n_neighbors': 15},
mean: 0.92817, std: 0.00830, params: {'n_neighbors': 16},
mean: 0.92559, std: 0.00873, params: {'n_neighbors': 17},
mean: 0.92728, std: 0.00821, params: {'n_neighbors': 18},
mean: 0.92497, std: 0.00821, params: {'n_neighbors': 19},
mean: 0.92470, std: 0.00830, params: {'n_neighbors': 20},
mean: 0.92204, std: 0.00804, params: {'n_neighbors': 21},
mean: 0.92186, std: 0.00867, params: {'n_neighbors': 22},
mean: 0.91902, std: 0.00856, params: {'n_neighbors': 23},
mean: 0.91777, std: 0.00850, params: {'n_neighbors': 24},
mean: 0.91564, std: 0.00809, params: {'n_neighbors': 25},
mean: 0.91457, std: 0.00913, params: {'n_neighbors': 26},
mean: 0.91244, std: 0.00927, params: {'n_neighbors': 27},
mean: 0.91244, std: 0.00873, params: {'n_neighbors': 28},
mean: 0.91101, std: 0.00878, params: {'n_neighbors': 29},
mean: 0.91110, std: 0.00853, params: {'n_neighbors': 30},
mean: 0.90906, std: 0.00855, params: {'n_neighbors': 31},
mean: 0.90941, std: 0.00906, params: {'n_neighbors': 32},
mean: 0.90773, std: 0.00922, params: {'n_neighbors': 33},
mean: 0.90728, std: 0.00991, params: {'n_neighbors': 34},
mean: 0.90550, std: 0.00946, params: {'n_neighbors': 35},
mean: 0.90417, std: 0.00891, params: {'n_neighbors': 36},
mean: 0.90292, std: 0.00887, params: {'n_neighbors': 37},
mean: 0.90097, std: 0.00881, params: {'n_neighbors': 38},
mean: 0.90035, std: 0.00948, params: {'n_neighbors': 39},
mean: 0.89875, std: 0.01009, params: {'n_neighbors': 40},
mean: 0.89759, std: 0.01009, params: {'n_neighbors': 41},
mean: 0.89626, std: 0.00955, params: {'n_neighbors': 42},
mean: 0.89537, std: 0.01016, params: {'n_neighbors': 43},
mean: 0.89270, std: 0.01221, params: {'n_neighbors': 44},
mean: 0.89172, std: 0.01244, params: {'n_neighbors': 45},
mean: 0.88950, std: 0.01089, params: {'n_neighbors': 46},
mean: 0.88906, std: 0.01150, params: {'n_neighbors': 47},
mean: 0.88666, std: 0.01137, params: {'n_neighbors': 48},
mean: 0.88577, std: 0.01143, params: {'n_neighbors': 49},
mean: 0.88523, std: 0.01132, params: {'n_neighbors': 50},
mean: 0.88452, std: 0.01149, params: {'n_neighbors': 51},
mean: 0.88372, std: 0.01237, params: {'n_neighbors': 52},
mean: 0.88283, std: 0.01227, params: {'n_neighbors': 53},
mean: 0.88141, std: 0.01166, params: {'n_neighbors': 54},
mean: 0.88114, std: 0.01085, params: {'n_neighbors': 55},
mean: 0.88061, std: 0.01041, params: {'n_neighbors': 56},
mean: 0.87990, std: 0.01105, params: {'n_neighbors': 57},
mean: 0.87910, std: 0.00974, params: {'n_neighbors': 58},
mean: 0.87892, std: 0.00997, params: {'n_neighbors': 59},
mean: 0.87803, std: 0.01108, params: {'n_neighbors': 60},
mean: 0.87768, std: 0.01119, params: {'n_neighbors': 61},
mean: 0.87688, std: 0.01152, params: {'n_neighbors': 62},
mean: 0.87652, std: 0.01182, params: {'n_neighbors': 63},
mean: 0.87590, std: 0.01129, params: {'n_neighbors': 64},
mean: 0.87554, std: 0.01148, params: {'n_neighbors': 65},
mean: 0.87474, std: 0.01183, params: {'n_neighbors': 66},
mean: 0.87412, std: 0.01223, params: {'n_neighbors': 67},
mean: 0.87394, std: 0.01196, params: {'n_neighbors': 68},
mean: 0.87341, std: 0.01210, params: {'n_neighbors': 69},
mean: 0.87314, std: 0.01231, params: {'n_neighbors': 70},
mean: 0.87270, std: 0.01219, params: {'n_neighbors': 71},
mean: 0.87226, std: 0.01168, params: {'n_neighbors': 72},
mean: 0.87154, std: 0.01185, params: {'n_neighbors': 73},
mean: 0.87074, std: 0.01104, params: {'n_neighbors': 74},
mean: 0.87057, std: 0.01132, params: {'n_neighbors': 75},
mean: 0.87003, std: 0.01129, params: {'n_neighbors': 76},
mean: 0.86968, std: 0.01148, params: {'n_neighbors': 77},
mean: 0.86754, std: 0.01056, params: {'n_neighbors': 78},
mean: 0.86692, std: 0.01041, params: {'n_neighbors': 79},
mean: 0.86568, std: 0.01028, params: {'n_neighbors': 80},
mean: 0.86594, std: 0.01024, params: {'n_neighbors': 81},
mean: 0.86461, std: 0.00937, params: {'n_neighbors': 82},
mean: 0.86390, std: 0.00915, params: {'n_neighbors': 83},
mean: 0.86239, std: 0.00956, params: {'n_neighbors': 84},
mean: 0.86257, std: 0.01001, params: {'n_neighbors': 85},
mean: 0.86088, std: 0.01023, params: {'n_neighbors': 86},
mean: 0.86088, std: 0.01051, params: {'n_neighbors': 87},
mean: 0.85981, std: 0.01214, params: {'n_neighbors': 88},
mean: 0.85937, std: 0.01208, params: {'n_neighbors': 89},
mean: 0.85830, std: 0.01087, params: {'n_neighbors': 90},
mean: 0.85821, std: 0.01108, params: {'n_neighbors': 91},
mean: 0.85616, std: 0.00995, params: {'n_neighbors': 92},
mean: 0.85652, std: 0.00954, params: {'n_neighbors': 93},
mean: 0.85519, std: 0.00892, params: {'n_neighbors': 94},
mean: 0.85545, std: 0.00957, params: {'n_neighbors': 95},
mean: 0.85368, std: 0.00900, params: {'n_neighbors': 96},
mean: 0.85332, std: 0.00845, params: {'n_neighbors': 97},
mean: 0.85199, std: 0.00923, params: {'n_neighbors': 98},
mean: 0.85190, std: 0.00926, params: {'n_neighbors': 99},
mean: 0.85021, std: 0.00825, params: {'n_neighbors': 100},
mean: 0.85039, std: 0.00930, params: {'n_neighbors': 101},
mean: 0.84834, std: 0.01015, params: {'n_neighbors': 102},
mean: 0.84825, std: 0.01016, params: {'n_neighbors': 103},
mean: 0.84559, std: 0.00954, params: {'n_neighbors': 104},
mean: 0.84559, std: 0.00922, params: {'n_neighbors': 105},
mean: 0.84416, std: 0.00977, params: {'n_neighbors': 106},
mean: 0.84452, std: 0.00978, params: {'n_neighbors': 107},
mean: 0.84372, std: 0.00935, params: {'n_neighbors': 108},
mean: 0.84399, std: 0.00922, params: {'n_neighbors': 109},
mean: 0.84256, std: 0.00943, params: {'n_neighbors': 110},
mean: 0.84239, std: 0.00930, params: {'n_neighbors': 111},
mean: 0.84132, std: 0.00941, params: {'n_neighbors': 112},
mean: 0.84087, std: 0.00924, params: {'n_neighbors': 113},
mean: 0.84034, std: 0.00864, params: {'n_neighbors': 114},
mean: 0.84016, std: 0.00875, params: {'n_neighbors': 115},
mean: 0.83981, std: 0.00849, params: {'n_neighbors': 116},
mean: 0.83972, std: 0.00862, params: {'n_neighbors': 117},
mean: 0.83865, std: 0.00779, params: {'n_neighbors': 118},
mean: 0.83892, std: 0.00782, params: {'n_neighbors': 119},
mean: 0.83883, std: 0.00798, params: {'n_neighbors': 120},
mean: 0.83892, std: 0.00810, params: {'n_neighbors': 121},
mean: 0.83661, std: 0.00577, params: {'n_neighbors': 122},
mean: 0.83705, std: 0.00582, params: {'n_neighbors': 123},
mean: 0.83670, std: 0.00582, params: {'n_neighbors': 124},
mean: 0.83714, std: 0.00545, params: {'n_neighbors': 125},
mean: 0.83732, std: 0.00565, params: {'n_neighbors': 126},
mean: 0.83741, std: 0.00610, params: {'n_neighbors': 127},
mean: 0.83732, std: 0.00647, params: {'n_neighbors': 128},
mean: 0.83741, std: 0.00634, params: {'n_neighbors': 129},
mean: 0.83759, std: 0.00626, params: {'n_neighbors': 130},
mean: 0.83750, std: 0.00609, params: {'n_neighbors': 131},
mean: 0.83759, std: 0.00639, params: {'n_neighbors': 132},
mean: 0.83767, std: 0.00616, params: {'n_neighbors': 133},
mean: 0.83732, std: 0.00640, params: {'n_neighbors': 134},
mean: 0.83776, std: 0.00641, params: {'n_neighbors': 135},
mean: 0.83723, std: 0.00690, params: {'n_neighbors': 136},
mean: 0.83794, std: 0.00675, params: {'n_neighbors': 137},
mean: 0.83687, std: 0.00710, params: {'n_neighbors': 138},
mean: 0.83687, std: 0.00718, params: {'n_neighbors': 139},
mean: 0.83607, std: 0.00744, params: {'n_neighbors': 140},
mean: 0.83625, std: 0.00725, params: {'n_neighbors': 141},
mean: 0.83501, std: 0.00739, params: {'n_neighbors': 142},
mean: 0.83536, std: 0.00761, params: {'n_neighbors': 143},
mean: 0.83430, std: 0.00774, params: {'n_neighbors': 144},
mean: 0.83536, std: 0.00776, params: {'n_neighbors': 145},
mean: 0.83510, std: 0.00746, params: {'n_neighbors': 146},
mean: 0.83527, std: 0.00732, params: {'n_neighbors': 147},
mean: 0.83412, std: 0.00716, params: {'n_neighbors': 148},
mean: 0.83492, std: 0.00704, params: {'n_neighbors': 149},
mean: 0.83474, std: 0.00740, params: {'n_neighbors': 150},
mean: 0.83483, std: 0.00759, params: {'n_neighbors': 151},
mean: 0.83492, std: 0.00751, params: {'n_neighbors': 152},
mean: 0.83536, std: 0.00788, params: {'n_neighbors': 153},
mean: 0.83492, std: 0.00833, params: {'n_neighbors': 154},
mean: 0.83492, std: 0.00841, params: {'n_neighbors': 155},
mean: 0.83492, std: 0.00815, params: {'n_neighbors': 156},
mean: 0.83510, std: 0.00822, params: {'n_neighbors': 157},
mean: 0.83527, std: 0.00826, params: {'n_neighbors': 158},
mean: 0.83554, std: 0.00832, params: {'n_neighbors': 159},
mean: 0.83554, std: 0.00778, params: {'n_neighbors': 160},
mean: 0.83519, std: 0.00745, params: {'n_neighbors': 161},
mean: 0.83519, std: 0.00757, params: {'n_neighbors': 162},
mean: 0.83536, std: 0.00715, params: {'n_neighbors': 163},
mean: 0.83527, std: 0.00767, params: {'n_neighbors': 164},
mean: 0.83536, std: 0.00778, params: {'n_neighbors': 165},
mean: 0.83492, std: 0.00759, params: {'n_neighbors': 166},
mean: 0.83510, std: 0.00790, params: {'n_neighbors': 167},
mean: 0.83527, std: 0.00820, params: {'n_neighbors': 168},
mean: 0.83492, std: 0.00775, params: {'n_neighbors': 169},
mean: 0.83456, std: 0.00746, params: {'n_neighbors': 170},
mean: 0.83447, std: 0.00782, params: {'n_neighbors': 171},
mean: 0.83376, std: 0.00770, params: {'n_neighbors': 172},
mean: 0.83376, std: 0.00731, params: {'n_neighbors': 173},
mean: 0.83270, std: 0.00737, params: {'n_neighbors': 174},
mean: 0.83287, std: 0.00736, params: {'n_neighbors': 175},
mean: 0.83261, std: 0.00701, params: {'n_neighbors': 176},
mean: 0.83234, std: 0.00676, params: {'n_neighbors': 177},
mean: 0.83225, std: 0.00719, params: {'n_neighbors': 178},
mean: 0.83207, std: 0.00718, params: {'n_neighbors': 179},
mean: 0.83181, std: 0.00746, params: {'n_neighbors': 180},
mean: 0.83190, std: 0.00741, params: {'n_neighbors': 181},
mean: 0.83243, std: 0.00808, params: {'n_neighbors': 182},
mean: 0.83261, std: 0.00827, params: {'n_neighbors': 183},
mean: 0.83225, std: 0.00819, params: {'n_neighbors': 184},
mean: 0.83279, std: 0.00779, params: {'n_neighbors': 185},
mean: 0.83118, std: 0.00688, params: {'n_neighbors': 186},
mean: 0.83154, std: 0.00684, params: {'n_neighbors': 187},
mean: 0.82950, std: 0.00716, params: {'n_neighbors': 188},
mean: 0.83003, std: 0.00689, params: {'n_neighbors': 189},
mean: 0.82870, std: 0.00649, params: {'n_neighbors': 190},
mean: 0.82878, std: 0.00704, params: {'n_neighbors': 191},
mean: 0.82665, std: 0.00624, params: {'n_neighbors': 192},
mean: 0.82718, std: 0.00654, params: {'n_neighbors': 193},
mean: 0.82478, std: 0.00895, params: {'n_neighbors': 194},
mean: 0.82550, std: 0.00846, params: {'n_neighbors': 195},
mean: 0.82230, std: 0.00974, params: {'n_neighbors': 196},
mean: 0.82274, std: 0.01041, params: {'n_neighbors': 197},
mean: 0.81936, std: 0.00926, params: {'n_neighbors': 198},
mean: 0.81972, std: 0.00963, params: {'n_neighbors': 199}]
In [104]:
# create a list of the mean scores only
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print(grid_mean_scores)
[0.9623966574806649, 0.9588407858476309, 0.9456840608054049, 0.94621744155036, 0.9387501111209885, 0.9428393634989777, 0.9379500400035559, 0.9383945239576851, 0.9335052004622633, 0.9349275491154769, 0.9304827095741843, 0.9325273357631789, 0.9293270512934483, 0.9296826384567517, 0.9271935283136279, 0.9281713930127122, 0.9255933860787625, 0.9272824251044537, 0.9249711085429816, 0.9247044181705041, 0.9220375144457285, 0.9218597208640769, 0.9190150235576495, 0.9177704684860877, 0.9156369455062672, 0.914570184016357, 0.9124366610365365, 0.9124366610365365, 0.911014312383323, 0.9111032091741488, 0.9090585829851542, 0.9094141701484576, 0.9077251311227664, 0.9072806471686372, 0.9055027113521202, 0.9041692594897324, 0.9029247044181705, 0.9009689750200017, 0.9003466974842208, 0.8987465552493555, 0.8975908969686194, 0.8962574451062316, 0.8953684771979732, 0.8927015734731976, 0.8917237087741132, 0.8895012890034669, 0.8890568050493377, 0.8866565916970397, 0.8857676237887813, 0.8852342430438261, 0.8845230687172193, 0.8837229975997867, 0.8828340296915281, 0.8814116810383145, 0.881144990665837, 0.8806116099208818, 0.879900435594275, 0.8791003644768424, 0.8789225708951907, 0.8780336029869322, 0.8776780158236288, 0.8768779447061961, 0.8765223575428926, 0.8759000800071117, 0.8755444928438083, 0.8747444217263757, 0.8741221441905948, 0.873944350608943, 0.8734109698639879, 0.8731442794915104, 0.8726997955373811, 0.8722553115832519, 0.871544137256645, 0.8707440661392124, 0.8705662725575607, 0.8700328918126056, 0.8696773046493022, 0.8675437816694818, 0.8669215041337007, 0.8656769490621389, 0.8659436394346164, 0.8646101875722286, 0.8638990132456218, 0.8623877678015823, 0.862565561383234, 0.8608765223575429, 0.8608765223575429, 0.8598097608676327, 0.8593652769135034, 0.8582985154235933, 0.8582096186327673, 0.8561649924437728, 0.8565205796070762, 0.8551871277446884, 0.8554538181171659, 0.853675882300649, 0.8533202951373455, 0.8519868432749578, 0.851897946484132, 0.8502089074584408, 0.8503867010400925, 0.8483420748510979, 0.848253178060272, 0.8455862743354965, 0.8455862743354965, 0.8441639256822828, 0.8445195128455862, 0.8437194417281536, 0.8439861321006311, 0.8425637834474176, 0.8423859898657658, 0.8413192283758556, 0.8408747444217264, 0.8403413636767713, 0.8401635700951195, 0.8398079829318161, 0.8397190861409903, 0.8386523246510801, 0.8389190150235577, 0.8388301182327318, 0.8389190150235577, 0.8366076984620855, 0.8370521824162148, 0.8366965952529114, 0.8371410792070406, 0.8373188727886923, 0.8374077695795181, 0.8373188727886923, 0.8374077695795181, 0.8375855631611698, 0.837496666370344, 0.8375855631611698, 0.8376744599519957, 0.8373188727886923, 0.8377633567428215, 0.8372299759978665, 0.8379411503244732, 0.8368743888345631, 0.8368743888345631, 0.8360743177171304, 0.8362521112987821, 0.8350075562272202, 0.8353631433905236, 0.8342963819006134, 0.8353631433905236, 0.835096453018046, 0.8352742465996977, 0.8341185883189617, 0.8349186594363943, 0.8347408658547426, 0.8348297626455685, 0.8349186594363943, 0.8353631433905236, 0.8349186594363943, 0.8349186594363943, 0.8349186594363943, 0.835096453018046, 0.8352742465996977, 0.8355409369721754, 0.8355409369721754, 0.8351853498088719, 0.8351853498088719, 0.8353631433905236, 0.8352742465996977, 0.8353631433905236, 0.8349186594363943, 0.835096453018046, 0.8352742465996977, 0.8349186594363943, 0.8345630722730909, 0.8344741754822651, 0.8337630011556583, 0.8337630011556583, 0.832696239665748, 0.8328740332473997, 0.8326073428749222, 0.8323406525024447, 0.8322517557116188, 0.8320739621299671, 0.8318072717574896, 0.8318961685483154, 0.8324295492932705, 0.8326073428749222, 0.8322517557116188, 0.8327851364565739, 0.8311849942217086, 0.831540581385012, 0.8294959551960174, 0.8300293359409725, 0.8286958840785847, 0.8287847808694107, 0.8266512578895902, 0.8271846386345453, 0.8247844252822473, 0.8254955996088541, 0.8222953151391235, 0.8227397990932528, 0.8193617210418704, 0.8197173082051737]
In [105]:
# plot the results (Manhattan distance)
plt.plot(k_potentials, grid_mean_scores)
plt.xlabel('Potential K values for KNN')
plt.ylabel('Cross-Validated Accuracy')
Out[105]:
<matplotlib.text.Text at 0x120cb5a50>

In [106]:
# plot the results (using Euclidean distance)
plt.plot(k_potentials, grid_mean_scores)
plt.xlabel('Potential K values for KNN')
plt.ylabel('Cross-Validated Accuracy')
Out[106]:
<matplotlib.text.Text at 0x12143b7d0>

In [107]:
# examine the best model
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
0.962396657481
{'n_neighbors': 1}
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
metric_params=None, n_jobs=1, n_neighbors=1, p=2,
weights='uniform')
In [ ]:

In [108]:
# read in the data & create matrices
X = X_scaled[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", \
"time_spend_company", "Work_accident", "promotion_last_5years", "low", "medium",\
"technical", "support", "IT", "product_mng", "marketing", "RandD", "accounting",\
"hr"]]
y = X_scaled["left"]
In [176]:
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
Support Vector Machines
In [177]:
# Train, fit and predict with SVM

from sklearn import svm
Model=svm.SVC(kernel='linear')
Model.fit(X_train,y_train)
Y_pred=Model.predict(X_test)
In [178]:
# Cross validation score for SVM
scores = cross_val_score(Model, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.780694712068
In [179]:
# Evaluate SVM for accuracy

from sklearn import metrics
metrics.accuracy_score(y_test,Y_pred)
Out[179]:
0.78106666666666669
In [180]:
# Confusion matrix for SVM

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2696  166]
[ 655  233]]
In [181]:
# Classification Accuracy confirmation between manual calculation and formula for SVM

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.781066666667
0.781066666667
In [182]:
# Misclassification Accuracy confirmation between manual calculation and formula for SVM
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.218933333333
0.218933333333
In [183]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated SVM

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.262387387387
0.262387387387
In [184]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for SVM

print(TN / float(TN + FP))
0.941998602376
In [185]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for SVM

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.583959899749
0.583959899749
In [186]:
# Misclassification Accuracy confirmation between manual calculation and formula for SVM
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.218933333333
0.218933333333
In [187]:
# Roc_curve for Support Vector Machine Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [92]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [93]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Support Vector Machines: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

Random Forest
In [157]:
# Random Forests Classifier from Scikit Learn

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_jobs=2)
RF.fit(X_train,y_train)
Y_pred=RF.predict(X_test)
In [158]:
# Score for 10-fold Cross Validation for Random Forests Classifier

scores = cross_val_score(RF, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.986309602501
In [159]:
# Prediction Accuracy for Random Forests

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.9896
In [160]:
# Confusion matrix for Random Forests

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2857    5]
[  34  854]]
In [161]:
# Classification Accuracy confirmation between manual calculation and formula for Random Forest

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.9896
0.9896
In [162]:
# Misclassification Accuracy confirmation between manual calculation and formula for Random Forest
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0104
0.0104
In [163]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for Random Forest Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.961711711712
0.961711711712
In [164]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for Random Forest Classifier

print(TN / float(TN + FP))
0.998252969951
In [165]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for Random Forest Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.994179278231
0.994179278231
In [166]:
# Misclassification Accuracy confirmation between manual calculation and formula for Random Forest
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0104
0.0104
In [167]:
# Roc_curve for Random Forest Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [168]:
y_pred_prob_Random_Forest = RF.predict_proba(X_test)[:, 1]
In [172]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_Random_Forest, bins=5)
plt.xlim(0, 1)
plt.title('Random Forest Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[172]:
<matplotlib.text.Text at 0x12013a8d0>

In [170]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [171]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Random Forest: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

Bagging
In [109]:
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
In [110]:
# Bagging Classifier

seed = 7
cart = DecisionTreeClassifier()
num_trees = 100
Bagging = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
Bagging.fit(X_train,y_train)
Y_pred=Bagging.predict(X_test)
In [111]:
# Cross Validation for Bagging

scores = cross_val_score(Bagging, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.987910076856
In [112]:
# Prediction Accuracy for Bagging

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.989333333333
In [113]:
# Confusion matrix for Bagging

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2854    8]
[  32  856]]
In [114]:
# Classification Accuracy confirmation between manual calculation and formula for Bagging

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.989333333333
0.989333333333
In [115]:
# Misclassification Accuracy confirmation between manual calculation and formula for Bagging
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0106666666667
0.0106666666667
In [116]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for Bagging Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.963963963964
0.963963963964
In [117]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for Bagging Classifier

print(TN / float(TN + FP))
0.997204751922
In [118]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for Bagging Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.990740740741
0.990740740741
In [119]:
# Misclassification Accuracy confirmation between manual calculation and formula for Bagging
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0106666666667
0.0106666666667
In [120]:
# Roc_curve for Bagging


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [121]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
In [122]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Bagging: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [123]:
y_pred_prob_Bagging = Bagging.predict_proba(X_test)[:, 1]
In [124]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_Bagging, bins=5)
plt.xlim(0, 1)
plt.title('Bagging Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[124]:
<matplotlib.text.Text at 0x11b217610>

In [ ]:

AdaBoost
In [125]:
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
In [126]:
# AdaBoost Classifier

seed = 7
cart = AdaBoostClassifier()
num_trees = 100
AdaBoostClassifier = AdaBoostClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
AdaBoostClassifier.fit(X_train,y_train)
Y_pred=AdaBoostClassifier.predict(X_test)
In [127]:
# Cross Validation for AdaBoost

scores = cross_val_score(AdaBoostClassifier, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.958576016468
In [128]:
# Prediction Accuracy for AdaBoost

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.958133333333
In [129]:
# Confusion matrix for AdaBoost

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2793   69]
[  88  800]]
In [130]:
# Classification Accuracy confirmation between manual calculation and formula for Bagging

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.958133333333
0.958133333333
In [131]:
# Misclassification Accuracy confirmation between manual calculation and formula for AdaBoost
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0418666666667
0.0418666666667
In [132]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for AdaBoost Boosting Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.900900900901
0.900900900901
In [133]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for AdaBoost Classifier

print(TN / float(TN + FP))
0.975890985325
In [134]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for AdaBoost Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.920598388953
0.920598388953
In [135]:
# Misclassification Accuracy confirmation between manual calculation and formula for AdaBoost
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0418666666667
0.0418666666667
In [136]:
# Roc_curve for AdaBoost


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [137]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
In [138]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='AdaBoost: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [139]:
y_pred_prob_AdaBoost = AdaBoostClassifier.predict_proba(X_test)[:, 1]
In [140]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_AdaBoost, bins=5)
plt.xlim(0, 1)
plt.title('AdaBoost Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[140]:
<matplotlib.text.Text at 0x11b367a10>

In [ ]:

In [ ]:

In [ ]:

Gradient Boosting
In [141]:
from sklearn.ensemble import GradientBoostingClassifier
In [142]:
# Train and predict Gradient Boosting

GradientBoosting = GradientBoostingClassifier()
GradientBoosting.fit(X_train,y_train)
Y_pred=GradientBoosting.predict(X_test)
In [143]:
# Cross Validation for Gradient Boosting

scores = cross_val_score(GradientBoosting, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.976443246546
In [144]:
# Prediction Accuracy for Gradient Boosting

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.974133333333
In [145]:
# Confusion matrix for Gradient Boosting

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2832   30]
[  67  821]]
In [146]:
# Classification Accuracy confirmation between manual calculation and formula for Gradient Boosting

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.974133333333
0.974133333333
In [147]:
# Misclassification Accuracy confirmation between manual calculation and formula for Gradient Boosting
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0258666666667
0.0258666666667
In [148]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for Gradient Boosting Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.92454954955
0.92454954955
In [149]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for Gradient Classifier

print(TN / float(TN + FP))
0.989517819706
In [150]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for Gradient Boosting Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.964747356052
0.964747356052
In [151]:
# Misclassification Accuracy confirmation between manual calculation and formula for Gradient Boosting
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0258666666667
0.0258666666667
In [152]:
# Roc_curve for Gradient Boosting


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Gradient Boosting Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [153]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
In [154]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Gradient Boosting: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [155]:
y_pred_prob_GradientBoosting = GradientBoosting.predict_proba(X_test)[:, 1]
In [156]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_GradientBoosting, bins=5)
plt.xlim(0, 1)
plt.title('Gradient Boosting Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[156]:
<matplotlib.text.Text at 0x11c2566d0>

In [ ]:

In [ ]:

Part IV
Conclusion
One interesting chart showed us which departments have the most people leaving (and staying). They are Sales followed by Technical and Support. That means that the biggest effort at the company should be placed on keeping the strong employees in those departments (but by no means neglect the other departments).
There are several other interesting patterns in the data. Some are very intuitive whereas others very surprising and unexpected. There are three clusters of people leaving the company (the most important cluster is not as dense, in the sense that there are actually slightly more people there that stay than leave, but nevertheless its the most interesting and surprising cluster).
The three groups are employees with high performance and low satisfaction, employees with low satisfcation and performance (rather obvious) and employees with high satisfaction and performance (a surprising result). The first two results are very strong but rather trivial, there is limited use in providing detailed analyses of them. However, the somewhat less dense but very significant group of highly satisfied employees that do remarkably well in their evaluations is of extreme importance. Why do they leave? Well, there are several important factors to consider.
The people with high salaries and at even average satisfaction levels rarely leave the firm. Hence, one attribute that people who tend to leave will generally possess is salaries that are not high and fall in the low or medium range. It is interesting to note though that people with near perfect to perfect satisfaction levels do not leave even if they are in the low or medium income range brackets. Hence, a possible course of action to potentially alleviate this issue is increase salaries for good employees. Also, a further investigation into what might be the differentiating factor between people who are very satisfied (0.7-0.9 rate) vs super satisfied (0.9-1.0) can be done to see if it is possible to further increase the satisfaction level of those that are very satisfied but not perfectly so (maybe its possible to do this while at the same time saving money by not instituting as much pay increases if there is a way to get them satisfied in another manner, perhaps by working less hours for instance) would be a good idea.
Secondly, promotions are a huge factor. Looking at the plot of promotions during the past five years, it is very clear that people who are promoted stay and people who are not promoted leave. Of course, promotions and salary increases have some correlation because people tend to receive salary increases during promotions. Hence, it's important to promote employees if they do good work! That will help decrease the turnover rate.
The third important point is that employees who work a lot of hours that stay at the company for at least a medium amount of time tend to leave. Only the relatively small number of employees who work at the company for many years stay at the company regardless of working hours but very few make it there. Hence, the hours worked plays an important decision making role when employees think about whether to stay at the firm or seek other employment. Due to this finding it is important to see if any kind of rebalancing can be done so that part of the work can perhaps be shifted between employees who do not work as many hours and those that do in order to make the distribution of hours worked by employees more uniform which can perhaps increase employees willingness to stay at the company longer. It is quiet possible that some employees just feel burnt out even if they are generally happy with their job. Due to an inefficient work life balance, they might seek to find employment elsewhere.
Multiple machine learning algorithms were ran on the data and quiet a few did relatively well in terms of making predictions on out of sample data as measured by the accuracy score, the confusion matrix and deriatives thereof as well as the ROC Curve. In particular, the Random Forest and Bagging algorithms did the best with accuracy and cross-validation scores marginally different from each other just under 99%. Both correctly predicted over 3,700 out of 3,750 employees accurately (compared to a baseline accuracy of only 76% (2,850) as to whether they would leave/stay). The various scores computed from the confusion matrix were all almost perfect. Particularly importantly, the ROC curve has an area covering of .98 for both those algorithms.
Hence, using either of these two algorithms (Random Forests or Bagging) have given us great predictions as to which employees will leave vs which ones would stay.
Assumptions
We have assumed the dataset is accurate as we have not audited its validity.
Another assumption we made is that we trained on 75% of the data, conducted cross-validation, and then tested on the 25% remaining out of sample. However, when trying to sample to train a larger sample of the data (such as 90/10 test/train split), we received very similar results in our algorithms. Hence, our model is robust as should perform well when used on future data.
An assumption made by the Random Forest Model which did exceptionally well is that the sampling that it conducted during boostrap aggregation is actually representative. This is a safe assumption to make and the model has done really well on out of sample data.
Introduction:
This project analyzes a dataset in an attempt to determine what causes the good employees to leave their current jobs. The first step is to clean the data (Part I) and visualize it (Part II) using various types of graphs and charts to help establish clear relationships between various features. Part III analyzes the dataset using various Machine Learning algorithms and picks the best algorithm to model this dataset based on various metrics. Lastly, a conclusion on the entire analysis is provided.
Part I
Data Cleaning
This initial stage of the project is to clean the data in such a way that all the important attributes of the dataset be retained and unhindered for analysis purposes. Hence, the initial cleaning phase of the project imports the dataset and the libraries needed for the cleaning stages of the project. A check for Null Values/Missing Values is conducted (none are present). The multiple categorical variables in the dataset are created into dummy variables which is necessary for later stages of the analysis. These new variables are afterwards added to the original dataset and the initial variables from which dummification was done are removed to avoid redundancy. Lastly, the dataset is normalized due to the need to apply machine learning algorithms to it in later stages of the project.
In [1]:
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

%matplotlib inline
In [2]:
HR = pd.read_csv('HR_comma_sep.csv')
HR.head(10)
Out[2]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary
0	0.38	0.53	2	157	3	0	1	0	sales	low
1	0.80	0.86	5	262	6	0	1	0	sales	medium
2	0.11	0.88	7	272	4	0	1	0	sales	medium
3	0.72	0.87	5	223	5	0	1	0	sales	low
4	0.37	0.52	2	159	3	0	1	0	sales	low
5	0.41	0.50	2	153	3	0	1	0	sales	low
6	0.10	0.77	6	247	4	0	1	0	sales	low
7	0.92	0.85	5	259	5	0	1	0	sales	low
8	0.89	1.00	5	224	5	0	1	0	sales	low
9	0.42	0.53	2	142	3	0	1	0	sales	low
In [3]:
HR[HR['satisfaction_level'] == np.nan]
HR[HR['last_evaluation'] == np.nan]
HR[HR['number_project'] == np.nan]
HR[HR['average_montly_hours'] == np.nan]
HR[HR['time_spend_company'] == np.nan]
HR[HR['Work_accident'] == np.nan]
HR[HR['left'] == np.nan]
HR[HR['sales'] == np.nan]
HR[HR['salary'] == np.nan]
Out[3]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary
In [4]:
# Create a heatmap, to check for null values.

plt.figure(figsize=(10,7))
sns.heatmap(HR.isnull(),yticklabels=False, cbar=False, cmap='viridis')
Out[4]:
<matplotlib.axes._subplots.AxesSubplot at 0x114c6ee50>

In [5]:
HR['sales'].value_counts()
Out[5]:
sales          4140
technical      2720
support        2229
IT             1227
product_mng     902
marketing       858
RandD           787
accounting      767
hr              739
management      630
Name: sales, dtype: int64
In [6]:
HR['salary'].value_counts()
Out[6]:
low       7316
medium    6446
high      1237
Name: salary, dtype: int64
In [7]:
HR['left'].value_counts()
Out[7]:
0    11428
1     3571
Name: left, dtype: int64
In [8]:
# Create dummy variables for all the departments

sales = pd.get_dummies(HR.sales)

sales.columns = ["IT", "RandD", "accounting", "hr", "management", "marketing", "product_mng", "sales_department", "support", "technical"]

sales.head()
Out[8]:
IT	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical
0	0	0	0	0	0	0	0	1	0	0
1	0	0	0	0	0	0	0	1	0	0
2	0	0	0	0	0	0	0	1	0	0
3	0	0	0	0	0	0	0	1	0	0
4	0	0	0	0	0	0	0	1	0	0
In [9]:
# Create dummy variables for the salary

salary = pd.get_dummies(HR.salary)

salary.columns = ['low', 'medium', 'high']

salary.head()
Out[9]:
low	medium	high
0	0	1	0
1	0	0	1
2	0	0	1
3	0	1	0
4	0	1	0
In [10]:
# Create dummy variable for stayed/left

left = pd.get_dummies(HR.left)

left.columns = ['left', 'stayed']

left.head(10)
Out[10]:
left	stayed
0	0	1
1	0	1
2	0	1
3	0	1
4	0	1
5	0	1
6	0	1
7	0	1
8	0	1
9	0	1
In [11]:
# Concatenate the original dataset with the newly created salary and sales dummified variables

HR = pd.concat([HR, salary, sales], axis = 1)

HR.head(10)
Out[11]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary	...	IT	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical
0	0.38	0.53	2	157	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
1	0.80	0.86	5	262	6	0	1	0	sales	medium	...	0	0	0	0	0	0	0	1	0	0
2	0.11	0.88	7	272	4	0	1	0	sales	medium	...	0	0	0	0	0	0	0	1	0	0
3	0.72	0.87	5	223	5	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
4	0.37	0.52	2	159	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
5	0.41	0.50	2	153	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
6	0.10	0.77	6	247	4	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
7	0.92	0.85	5	259	5	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
8	0.89	1.00	5	224	5	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
9	0.42	0.53	2	142	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
10 rows × 23 columns
In [12]:
HR.dtypes
Out[12]:
satisfaction_level       float64
last_evaluation          float64
number_project             int64
average_montly_hours       int64
time_spend_company         int64
Work_accident              int64
left                       int64
promotion_last_5years      int64
sales                     object
salary                    object
low                        uint8
medium                     uint8
high                       uint8
IT                         uint8
RandD                      uint8
accounting                 uint8
hr                         uint8
management                 uint8
marketing                  uint8
product_mng                uint8
sales_department           uint8
support                    uint8
technical                  uint8
dtype: object
In [13]:
# Create a new column 

salary_map = {'low': 1, 'medium': 2, 'high': 3}

HR['salary_variable'] = HR['salary'].apply(lambda x: salary_map[x])

HR.head()
Out[13]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary	...	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical	salary_variable
0	0.38	0.53	2	157	3	0	1	0	sales	low	...	0	0	0	0	0	0	1	0	0	1
1	0.80	0.86	5	262	6	0	1	0	sales	medium	...	0	0	0	0	0	0	1	0	0	2
2	0.11	0.88	7	272	4	0	1	0	sales	medium	...	0	0	0	0	0	0	1	0	0	2
3	0.72	0.87	5	223	5	0	1	0	sales	low	...	0	0	0	0	0	0	1	0	0	1
4	0.37	0.52	2	159	3	0	1	0	sales	low	...	0	0	0	0	0	0	1	0	0	1
5 rows × 24 columns
In [14]:
del HR['salary']

del HR['sales']
In [15]:
HR.head()
Out[15]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	low	medium	...	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical	salary_variable
0	0.38	0.53	2	157	3	0	1	0	0	1	...	0	0	0	0	0	0	1	0	0	1
1	0.80	0.86	5	262	6	0	1	0	0	0	...	0	0	0	0	0	0	1	0	0	2
2	0.11	0.88	7	272	4	0	1	0	0	0	...	0	0	0	0	0	0	1	0	0	2
3	0.72	0.87	5	223	5	0	1	0	0	1	...	0	0	0	0	0	0	1	0	0	1
4	0.37	0.52	2	159	3	0	1	0	0	1	...	0	0	0	0	0	0	1	0	0	1
5 rows × 22 columns
In [16]:
HR.dtypes
Out[16]:
satisfaction_level       float64
last_evaluation          float64
number_project             int64
average_montly_hours       int64
time_spend_company         int64
Work_accident              int64
left                       int64
promotion_last_5years      int64
low                        uint8
medium                     uint8
high                       uint8
IT                         uint8
RandD                      uint8
accounting                 uint8
hr                         uint8
management                 uint8
marketing                  uint8
product_mng                uint8
sales_department           uint8
support                    uint8
technical                  uint8
salary_variable            int64
dtype: object
In [17]:
from sklearn.preprocessing import MinMaxScaler
X_scaled = MinMaxScaler().fit_transform(HR.astype(float))
X_scaled = pd.DataFrame(X_scaled, columns = HR.columns)

# Min Max Scaler

X_scaled.head(10)
Out[17]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	low	medium	...	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical	salary_variable
0	0.318681	0.265625	0.0	0.285047	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
1	0.780220	0.781250	0.6	0.775701	0.500	0.0	1.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.5
2	0.021978	0.812500	1.0	0.822430	0.250	0.0	1.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.5
3	0.692308	0.796875	0.6	0.593458	0.375	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
4	0.307692	0.250000	0.0	0.294393	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
5	0.351648	0.218750	0.0	0.266355	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
6	0.010989	0.640625	0.8	0.705607	0.250	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
7	0.912088	0.765625	0.6	0.761682	0.375	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
8	0.879121	1.000000	0.6	0.598131	0.375	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
9	0.362637	0.265625	0.0	0.214953	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
10 rows × 22 columns
Part II
Exploratory Data Analysis (EDA)
In this section, we try to visualize the data on employees using various plots, graphs, charts in order to get an idea of how different features in the dataset relate to each other.
The below represents a heatmap of correlations of features. Some interesting observations from this heatmap: Quiet intuitively, you can see that the relationship between satisfaction level and people that left the company is negative. It's also interesting to note that there is a positive correlation between the number of hours employees work and their evaluation. These and other relationships will be explored more deeply in the rest of the EDA.
In [18]:
correlation = X_scaled.corr()
plt.figure(figsize=(20,20))
sns.heatmap(correlation, vmax=10, square=True,annot=True,cmap=None, linewidths=1)

plt.title('Correlation between features')
Out[18]:
<matplotlib.text.Text at 0x118f3a050>

In [19]:
# Exported csv file to work with in Tableau
X_scaled.to_csv("Updated_Capstone.csv")
In [20]:
X_scaled['left'].value_counts()
Out[20]:
0.0    11428
1.0     3571
Name: left, dtype: int64
In [21]:
stayed = np.where(X_scaled['left'] == 0)[0]
left = np.where(X_scaled['left'] == 1)[0]
For a given level of satisfaction, there is definitely an intuitive pattern that the higher someone's salary is, the less they are likely to leave. This relationship is a lot more apparent in the distinction between the high income earners vs low & median income earners as a group. The difference in the people that left between low and median income salaries is not significant. Hence, most of the people that leave are not receiving a high salary, which means that you if you would like to retain these employees, you should increase their pay!
In [22]:
# plot of Satisfaction vs Last Evaluation
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.satisfaction_level[stayed], X_scaled.salary_variable[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.satisfaction_level[left], X_scaled.salary_variable[left], alpha=0.1, color='r')
ax.set_title("satisfaction_level vs salary")
ax.set_xlabel("satisfaction_level")
ax.set_ylabel("salary")
plt.show()

In the Satisfaction vs Last Evaluation plot below, we can note some interesting observations. Firstly, there is a sizeable cluster of people with strong evaluations and low satisfaction levels leaving. Secondly, there is a significant group of people leaving that are slighly below average in performance and evaluation. Lastly, and by far the most interesting observation of the three, there is a decent size of a not very dense cluster of people leaving that have done extremely well on their evaluations (0.7 to 1 on a 0-1 scale) and and have very high satisfaction rates, albeit not perfect (0.7-0.9 on a 0-1 scale). That particular group of employees also have many people that stay, but a significant proportion of that group is leaving. So the most surprising thing to observe is that people can be satisfied with the job and still leave! There are several possibilites for why this might be possible. People might enjoy their job but think they can do even better elsewhere, whether its income or career growth (or a combination of the two, considering that those things tend to go hand in hand). Another possibility is that people might not be honest in surveys because of fear of being reprimanded for negative surveys.
In [23]:
# plot of Satisfaction vs Last Evaluation
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.satisfaction_level[stayed], X_scaled.last_evaluation[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.satisfaction_level[left], X_scaled.last_evaluation[left], alpha=0.1, color='r')
ax.set_title("satisfaction_level vs last_evaluation")
ax.set_xlabel("satisfaction_level")
ax.set_ylabel("last_evaluation")
plt.show()

The Satisfaction vs Promotion during the last five years plot below shows a very strong relationship in several interesting ways! People who have been promoted don't leave very often! However, such promotions are not common enough! Therefore, if you don't want good employees to leave, promoting them will really help!
In [24]:
# plot of Satisfaction vs Promotion during last five years
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.satisfaction_level[stayed], X_scaled.promotion_last_5years[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.satisfaction_level[left], X_scaled.promotion_last_5years[left], alpha=0.1, color='r')
ax.set_title("satisfaction_level vs promotion_last_5years")
ax.set_xlabel("satisfaction_level")
ax.set_ylabel("promotion_last_5years")
plt.show()

The below scatter plot compares the average working hours against the time spent at the company and how it impacts the employee turnover rate. There are multiple interesting relationships to note. For employees that just started out, the majority tend to stick around for some time even if they work substantial hours. However, for employees who have been at the company for a while, there is a significant tendency to leave the company if working substantial hours. Consequently, we can observe that very few employees stick around at the company for a long time period because the ones who were working long hours almost all left! However, those that do stick around for a long time (relatively few) tend to all be devoted and not leave
There are several conjectures for why this is possible. One is that employees have worked for a significant amount of hours and put in a lot of time and effort expecting a quick promotion and/or pay increase (refer to graph above for relationship between promotions and staying/leaving) that they have not received and leave for elsewhere. Another reason might be that people have tried working for a little bit of time, realized that they are being overworked and have started looking for other opportunities to seek employment.
In [25]:
# plot of Average_monthly_hours vs time_spend_company
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.average_montly_hours[stayed], X_scaled.time_spend_company[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.average_montly_hours[left], X_scaled.time_spend_company[left], alpha=0.1, color='r')
ax.set_title("average_monthly_hours vs time_spend_company")
ax.set_xlabel("average_monthly_hours")
ax.set_ylabel("time_spend_company")
plt.show()

In [26]:
# Imported from Tableau
from IPython.display import Image
Image(filename = 'Average Monthly Hours vs Satisfaction Level.png', width = 1000)
Out[26]:

There does not appear to be a strong relationship between accidents and people leaving the company, hence, people stay/leave regardless of the accidents they might have experienced.
In [27]:
# plot of work_accident vs time_spend_company
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.Work_accident[stayed], X_scaled.time_spend_company[stayed], alpha=0.5, color='g')
ax.scatter(X_scaled.Work_accident[left], X_scaled.time_spend_company[left], alpha=0.5, color='r')
ax.set_title("Work_accident vs time_spend_company")
ax.set_xlabel("Work_accident")
ax.set_ylabel("time_spend_company")
plt.show()

The below stripplot compares number of projects done by an employee versus the number of promotions they have received over the past 5 years. It is interesting to note that the number of projects completed has very insignificant impact on whether a person leaves. On the contrary, the promotions received during the past 5 years is of critical importance. The majority of people who have received a promotion stay. Almost all those who were not promoted leave.
The data seems insufficient to be able to determine why the number of projects is not related to promotions. There are at least several possibilities. Perhaps the number of projects refers to the number assigned (vs completed) or that the projects are not all of equal difficulty and not equally time consuming leading to distorted hypothesis that the number of projects might necessarily correspond to an employee's productivity.
In [28]:
import seaborn as sns

sns.stripplot("number_project", "promotion_last_5years", data=X_scaled, hue="left", jitter = True)
Out[28]:
<matplotlib.axes._subplots.AxesSubplot at 0x117949d90>

The Satisfaction level distribution below shows that the majority of people are at least moderately satisfied with their job. Therefore, for the people that are satisfied and do well, efforts should be placed on making them even more happy by promoting them. For the people that do poorly or mediocre, either studies should be placed on determining why they are not doing well and trying to improve their performance or attempting to replace them with better employees (this would depend on a cost/benefit analysis) done by the company regarding this issue.
In [29]:
# Histogram distribution of Satisfaction Level

f,ax1 = plt.subplots(1,1)
sns.distplot(X_scaled['satisfaction_level'],bins=10,norm_hist=False)
plt.show()

The distribution (which is a decent approximation of a normal distribution with almost no tails and some kurtosis present) of the last evaluation graph shows us there is a sizeable portion of people that do well in their evaluations, hence the company should try to do as much as possible to keep those people!
In [30]:
# Histogram distribution of Last Evaluation


f,ax1 = plt.subplots(1,1)
sns.distplot(X_scaled['last_evaluation'],bins=10,norm_hist=False)
plt.show()

The below plot shows the ratios of people who stay in the company by department and the ratio of people who leave the company by department. It is clear from the chart that the departments with the most number of employees also have the highest ratios of both people who have left and those that stayed at the company. This tells us there should be particular effort place on retaining good employees that work in those departments, in particular the sales, technical and support departments, respectively. The largest efforts should be placed on retaining the good employees in these groups because they have the most people leaving.
In [34]:
# This plot is to compare the people that left across different departments
# The non-department variables are deleted to only keep the departments

del X_scaled['satisfaction_level']
del X_scaled['last_evaluation']
del X_scaled['number_project']
del X_scaled['average_montly_hours']
del X_scaled['time_spend_company']
del X_scaled['Work_accident']
del X_scaled['promotion_last_5years']
del X_scaled['low']
del X_scaled['medium']
del X_scaled['high']
del X_scaled['salary_variable']


left=X_scaled.groupby('left').mean()
left=left.transpose()
plt.figure(figsize=(20,10))
left.plot(kind='bar')
Out[34]:
<matplotlib.axes._subplots.AxesSubplot at 0x11a7bcd10>
<matplotlib.figure.Figure at 0x11a360850>

Part III
Predictive Modelling
In this section below, multiple models will be run that are applicable to classification. Afterwards, various scoring metrics will be calculated for the models and a determination will be made as to which model is based for this dataset.
In [174]:
# read in the data & create matrices
X = X_scaled[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", \
"time_spend_company", "Work_accident", "promotion_last_5years", "salary_variable",\
"technical", "support", "IT", "product_mng", "marketing", "RandD", "accounting",\
"hr", "management", "sales_department"]]
y = X_scaled["left"]
In [175]:
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
Logistic Regression Model
In [33]:
# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
model = LogisticRegression()

# fit the model with data
mdl = model.fit(X_train, y_train)

# predict the response for new observations

logit = model.predict(X_test)

# sklearn output to check that ran on right data split.

len(logit)
Out[33]:
3750
In [34]:
# These represent the y-intercept and coefficients for all the variables in the logistic regression.

print(model.intercept_)
print(model.coef_)
[ 0.57305398]
[[-3.71487625  0.45484967 -1.50619896  0.8704181   1.97590531 -1.45425383
-1.26024249 -1.32512177  0.22588571  0.23497924 -0.01282827  0.07509077
0.13925062 -0.47542127  0.26606552  0.3920348  -0.44451555  0.17251241]]
In [35]:
# Prediction Accuracy for Logistic Regression

from sklearn import metrics
print(metrics.accuracy_score(y_test, logit))
0.796
In [36]:
from sklearn.cross_validation import cross_val_score
In [37]:
# 10-fold cross-validation for Logistic Regression to find the CV score
mdl = model.fit(X_train, y_train)
scores = cross_val_score(mdl, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.794205514996
In [38]:
from sklearn import metrics
confusion = metrics.confusion_matrix(y_test, logit)
print ((confusion))
[[2651  211]
[ 554  334]]
In [39]:
y_pred_prob = model.predict_proba(X_test)[:, 1]
In [40]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob, bins=5)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[40]:
<matplotlib.text.Text at 0x1181a6a90>

In [41]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [42]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, logit)


plot_confusion_matrix(cm, title='LogisticRegression: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [43]:
# Roc_curve for Logistic Regression Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, logit)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Logistic Regression Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

K Nearest Nearbours Model
K=1
In [44]:
from sklearn.neighbors import KNeighborsClassifier
In [45]:
# K-Nearest Neighbours with K=1
knn = KNeighborsClassifier(n_neighbors=1, metric = 'euclidean')
In [46]:
y = knn.fit(X_train, y_train)

KNN = knn.predict(X_test)

# sklearn output to check that ran on right data split.

len(KNN)
Out[46]:
3750
In [47]:
# Cross fold validation for K=1

md2 = knn.fit(X_train, y_train)
scores = cross_val_score(md2, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.962043001151
In [48]:
# Prediction Accuracy.

from sklearn import metrics
print(metrics.accuracy_score(y_test, KNN))
0.967466666667
In [49]:
confusion = metrics.confusion_matrix(y_test, KNN)
print confusion
[[2775   87]
[  35  853]]
In [50]:
# Classification Accuracy confirmation between manual calculation and formula for K=1

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, KNN))
0.967466666667
0.967466666667
In [51]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=1
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN))
0.0325333333333
0.0325333333333
In [52]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for K=1

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, KNN))
0.960585585586
0.960585585586
In [53]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for K=1


print(TN / float(TN + FP))
0.969601677149
In [54]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for K=1

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, KNN))
0.907446808511
0.907446808511
In [55]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=1
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN))
0.0325333333333
0.0325333333333
In [56]:
y_pred_prob_knn = knn.predict_proba(X_test)[:, 1]
In [57]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_knn, bins=5)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[57]:
<matplotlib.text.Text at 0x1181d6b90>

In [58]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [59]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, KNN)


plot_confusion_matrix(cm, title='KNN for K=1: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [60]:
# Roc_curve for KNN=1 Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, KNN)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('KNN=1 Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

K=5
In [61]:
# K-Nearest Neighbours with K=5

knn = KNeighborsClassifier(n_neighbors=5, metric = 'euclidean')
y = knn.fit(X_train, y_train)
KNN5 = knn.predict(X_test)

# sklearn output to check that ran on right data split.

len(KNN5)
Out[61]:
3750
In [62]:
# Cross Validation for K=5 Cross-Fold Validation

md2 = knn.fit(X_train, y_train)
scores = cross_val_score(md2, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.938309179837
In [63]:
# Prediction Accuracy on KNN=5 for accuracy score

from sklearn import metrics
print(metrics.accuracy_score(y_test, KNN5))
0.941333333333
In [64]:
confusion = metrics.confusion_matrix(y_test, KNN5)
print confusion
[[2731  131]
[  89  799]]
In [65]:
# Classification Accuracy confirmation between manual calculation and formula for K=5

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, KNN5))
0.941333333333
0.941333333333
In [66]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=5
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN5))
0.0586666666667
0.0586666666667
In [67]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for K=5

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, KNN5))
0.899774774775
0.899774774775
In [68]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for K=5

print(TN / float(TN + FP))
0.954227812718
In [69]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for K=5

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, KNN5))
0.859139784946
0.859139784946
In [70]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=5
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN5))
0.0586666666667
0.0586666666667
In [71]:
# Roc_curve for KNN=5 Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, KNN5)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [72]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [73]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, KNN5)


plot_confusion_matrix(cm, title='KNN for N=5: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [74]:
y_pred_prob_knn = knn.predict_proba(X_test)[:, 1]
In [75]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_knn, bins=5)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[75]:
<matplotlib.text.Text at 0x11acdea10>

In [ ]:

Grid Search for optimal N value in KNN
In [101]:
from sklearn.grid_search import GridSearchCV
/Users/Misha/anaconda/lib/python2.7/site-packages/sklearn/grid_search.py:43: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.
DeprecationWarning)
In [102]:
k_potentials = list(range(1, 200))
print(k_potentials)
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]
In [103]:
# Set Nearest Neighbours equal to k
knn = KNeighborsClassifier(n_neighbors=k_potentials, metric = 'euclidean')
# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_potentials)
# instantiate the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
# fit the grid with data
grid.fit(X_train, y_train)
# view the results
grid.grid_scores_
Out[103]:
[mean: 0.96240, std: 0.00591, params: {'n_neighbors': 1},
mean: 0.95884, std: 0.00543, params: {'n_neighbors': 2},
mean: 0.94568, std: 0.00463, params: {'n_neighbors': 3},
mean: 0.94622, std: 0.00555, params: {'n_neighbors': 4},
mean: 0.93875, std: 0.00620, params: {'n_neighbors': 5},
mean: 0.94284, std: 0.00673, params: {'n_neighbors': 6},
mean: 0.93795, std: 0.00746, params: {'n_neighbors': 7},
mean: 0.93839, std: 0.00819, params: {'n_neighbors': 8},
mean: 0.93351, std: 0.00932, params: {'n_neighbors': 9},
mean: 0.93493, std: 0.00848, params: {'n_neighbors': 10},
mean: 0.93048, std: 0.00825, params: {'n_neighbors': 11},
mean: 0.93253, std: 0.00810, params: {'n_neighbors': 12},
mean: 0.92933, std: 0.00898, params: {'n_neighbors': 13},
mean: 0.92968, std: 0.00824, params: {'n_neighbors': 14},
mean: 0.92719, std: 0.00747, params: {'n_neighbors': 15},
mean: 0.92817, std: 0.00830, params: {'n_neighbors': 16},
mean: 0.92559, std: 0.00873, params: {'n_neighbors': 17},
mean: 0.92728, std: 0.00821, params: {'n_neighbors': 18},
mean: 0.92497, std: 0.00821, params: {'n_neighbors': 19},
mean: 0.92470, std: 0.00830, params: {'n_neighbors': 20},
mean: 0.92204, std: 0.00804, params: {'n_neighbors': 21},
mean: 0.92186, std: 0.00867, params: {'n_neighbors': 22},
mean: 0.91902, std: 0.00856, params: {'n_neighbors': 23},
mean: 0.91777, std: 0.00850, params: {'n_neighbors': 24},
mean: 0.91564, std: 0.00809, params: {'n_neighbors': 25},
mean: 0.91457, std: 0.00913, params: {'n_neighbors': 26},
mean: 0.91244, std: 0.00927, params: {'n_neighbors': 27},
mean: 0.91244, std: 0.00873, params: {'n_neighbors': 28},
mean: 0.91101, std: 0.00878, params: {'n_neighbors': 29},
mean: 0.91110, std: 0.00853, params: {'n_neighbors': 30},
mean: 0.90906, std: 0.00855, params: {'n_neighbors': 31},
mean: 0.90941, std: 0.00906, params: {'n_neighbors': 32},
mean: 0.90773, std: 0.00922, params: {'n_neighbors': 33},
mean: 0.90728, std: 0.00991, params: {'n_neighbors': 34},
mean: 0.90550, std: 0.00946, params: {'n_neighbors': 35},
mean: 0.90417, std: 0.00891, params: {'n_neighbors': 36},
mean: 0.90292, std: 0.00887, params: {'n_neighbors': 37},
mean: 0.90097, std: 0.00881, params: {'n_neighbors': 38},
mean: 0.90035, std: 0.00948, params: {'n_neighbors': 39},
mean: 0.89875, std: 0.01009, params: {'n_neighbors': 40},
mean: 0.89759, std: 0.01009, params: {'n_neighbors': 41},
mean: 0.89626, std: 0.00955, params: {'n_neighbors': 42},
mean: 0.89537, std: 0.01016, params: {'n_neighbors': 43},
mean: 0.89270, std: 0.01221, params: {'n_neighbors': 44},
mean: 0.89172, std: 0.01244, params: {'n_neighbors': 45},
mean: 0.88950, std: 0.01089, params: {'n_neighbors': 46},
mean: 0.88906, std: 0.01150, params: {'n_neighbors': 47},
mean: 0.88666, std: 0.01137, params: {'n_neighbors': 48},
mean: 0.88577, std: 0.01143, params: {'n_neighbors': 49},
mean: 0.88523, std: 0.01132, params: {'n_neighbors': 50},
mean: 0.88452, std: 0.01149, params: {'n_neighbors': 51},
mean: 0.88372, std: 0.01237, params: {'n_neighbors': 52},
mean: 0.88283, std: 0.01227, params: {'n_neighbors': 53},
mean: 0.88141, std: 0.01166, params: {'n_neighbors': 54},
mean: 0.88114, std: 0.01085, params: {'n_neighbors': 55},
mean: 0.88061, std: 0.01041, params: {'n_neighbors': 56},
mean: 0.87990, std: 0.01105, params: {'n_neighbors': 57},
mean: 0.87910, std: 0.00974, params: {'n_neighbors': 58},
mean: 0.87892, std: 0.00997, params: {'n_neighbors': 59},
mean: 0.87803, std: 0.01108, params: {'n_neighbors': 60},
mean: 0.87768, std: 0.01119, params: {'n_neighbors': 61},
mean: 0.87688, std: 0.01152, params: {'n_neighbors': 62},
mean: 0.87652, std: 0.01182, params: {'n_neighbors': 63},
mean: 0.87590, std: 0.01129, params: {'n_neighbors': 64},
mean: 0.87554, std: 0.01148, params: {'n_neighbors': 65},
mean: 0.87474, std: 0.01183, params: {'n_neighbors': 66},
mean: 0.87412, std: 0.01223, params: {'n_neighbors': 67},
mean: 0.87394, std: 0.01196, params: {'n_neighbors': 68},
mean: 0.87341, std: 0.01210, params: {'n_neighbors': 69},
mean: 0.87314, std: 0.01231, params: {'n_neighbors': 70},
mean: 0.87270, std: 0.01219, params: {'n_neighbors': 71},
mean: 0.87226, std: 0.01168, params: {'n_neighbors': 72},
mean: 0.87154, std: 0.01185, params: {'n_neighbors': 73},
mean: 0.87074, std: 0.01104, params: {'n_neighbors': 74},
mean: 0.87057, std: 0.01132, params: {'n_neighbors': 75},
mean: 0.87003, std: 0.01129, params: {'n_neighbors': 76},
mean: 0.86968, std: 0.01148, params: {'n_neighbors': 77},
mean: 0.86754, std: 0.01056, params: {'n_neighbors': 78},
mean: 0.86692, std: 0.01041, params: {'n_neighbors': 79},
mean: 0.86568, std: 0.01028, params: {'n_neighbors': 80},
mean: 0.86594, std: 0.01024, params: {'n_neighbors': 81},
mean: 0.86461, std: 0.00937, params: {'n_neighbors': 82},
mean: 0.86390, std: 0.00915, params: {'n_neighbors': 83},
mean: 0.86239, std: 0.00956, params: {'n_neighbors': 84},
mean: 0.86257, std: 0.01001, params: {'n_neighbors': 85},
mean: 0.86088, std: 0.01023, params: {'n_neighbors': 86},
mean: 0.86088, std: 0.01051, params: {'n_neighbors': 87},
mean: 0.85981, std: 0.01214, params: {'n_neighbors': 88},
mean: 0.85937, std: 0.01208, params: {'n_neighbors': 89},
mean: 0.85830, std: 0.01087, params: {'n_neighbors': 90},
mean: 0.85821, std: 0.01108, params: {'n_neighbors': 91},
mean: 0.85616, std: 0.00995, params: {'n_neighbors': 92},
mean: 0.85652, std: 0.00954, params: {'n_neighbors': 93},
mean: 0.85519, std: 0.00892, params: {'n_neighbors': 94},
mean: 0.85545, std: 0.00957, params: {'n_neighbors': 95},
mean: 0.85368, std: 0.00900, params: {'n_neighbors': 96},
mean: 0.85332, std: 0.00845, params: {'n_neighbors': 97},
mean: 0.85199, std: 0.00923, params: {'n_neighbors': 98},
mean: 0.85190, std: 0.00926, params: {'n_neighbors': 99},
mean: 0.85021, std: 0.00825, params: {'n_neighbors': 100},
mean: 0.85039, std: 0.00930, params: {'n_neighbors': 101},
mean: 0.84834, std: 0.01015, params: {'n_neighbors': 102},
mean: 0.84825, std: 0.01016, params: {'n_neighbors': 103},
mean: 0.84559, std: 0.00954, params: {'n_neighbors': 104},
mean: 0.84559, std: 0.00922, params: {'n_neighbors': 105},
mean: 0.84416, std: 0.00977, params: {'n_neighbors': 106},
mean: 0.84452, std: 0.00978, params: {'n_neighbors': 107},
mean: 0.84372, std: 0.00935, params: {'n_neighbors': 108},
mean: 0.84399, std: 0.00922, params: {'n_neighbors': 109},
mean: 0.84256, std: 0.00943, params: {'n_neighbors': 110},
mean: 0.84239, std: 0.00930, params: {'n_neighbors': 111},
mean: 0.84132, std: 0.00941, params: {'n_neighbors': 112},
mean: 0.84087, std: 0.00924, params: {'n_neighbors': 113},
mean: 0.84034, std: 0.00864, params: {'n_neighbors': 114},
mean: 0.84016, std: 0.00875, params: {'n_neighbors': 115},
mean: 0.83981, std: 0.00849, params: {'n_neighbors': 116},
mean: 0.83972, std: 0.00862, params: {'n_neighbors': 117},
mean: 0.83865, std: 0.00779, params: {'n_neighbors': 118},
mean: 0.83892, std: 0.00782, params: {'n_neighbors': 119},
mean: 0.83883, std: 0.00798, params: {'n_neighbors': 120},
mean: 0.83892, std: 0.00810, params: {'n_neighbors': 121},
mean: 0.83661, std: 0.00577, params: {'n_neighbors': 122},
mean: 0.83705, std: 0.00582, params: {'n_neighbors': 123},
mean: 0.83670, std: 0.00582, params: {'n_neighbors': 124},
mean: 0.83714, std: 0.00545, params: {'n_neighbors': 125},
mean: 0.83732, std: 0.00565, params: {'n_neighbors': 126},
mean: 0.83741, std: 0.00610, params: {'n_neighbors': 127},
mean: 0.83732, std: 0.00647, params: {'n_neighbors': 128},
mean: 0.83741, std: 0.00634, params: {'n_neighbors': 129},
mean: 0.83759, std: 0.00626, params: {'n_neighbors': 130},
mean: 0.83750, std: 0.00609, params: {'n_neighbors': 131},
mean: 0.83759, std: 0.00639, params: {'n_neighbors': 132},
mean: 0.83767, std: 0.00616, params: {'n_neighbors': 133},
mean: 0.83732, std: 0.00640, params: {'n_neighbors': 134},
mean: 0.83776, std: 0.00641, params: {'n_neighbors': 135},
mean: 0.83723, std: 0.00690, params: {'n_neighbors': 136},
mean: 0.83794, std: 0.00675, params: {'n_neighbors': 137},
mean: 0.83687, std: 0.00710, params: {'n_neighbors': 138},
mean: 0.83687, std: 0.00718, params: {'n_neighbors': 139},
mean: 0.83607, std: 0.00744, params: {'n_neighbors': 140},
mean: 0.83625, std: 0.00725, params: {'n_neighbors': 141},
mean: 0.83501, std: 0.00739, params: {'n_neighbors': 142},
mean: 0.83536, std: 0.00761, params: {'n_neighbors': 143},
mean: 0.83430, std: 0.00774, params: {'n_neighbors': 144},
mean: 0.83536, std: 0.00776, params: {'n_neighbors': 145},
mean: 0.83510, std: 0.00746, params: {'n_neighbors': 146},
mean: 0.83527, std: 0.00732, params: {'n_neighbors': 147},
mean: 0.83412, std: 0.00716, params: {'n_neighbors': 148},
mean: 0.83492, std: 0.00704, params: {'n_neighbors': 149},
mean: 0.83474, std: 0.00740, params: {'n_neighbors': 150},
mean: 0.83483, std: 0.00759, params: {'n_neighbors': 151},
mean: 0.83492, std: 0.00751, params: {'n_neighbors': 152},
mean: 0.83536, std: 0.00788, params: {'n_neighbors': 153},
mean: 0.83492, std: 0.00833, params: {'n_neighbors': 154},
mean: 0.83492, std: 0.00841, params: {'n_neighbors': 155},
mean: 0.83492, std: 0.00815, params: {'n_neighbors': 156},
mean: 0.83510, std: 0.00822, params: {'n_neighbors': 157},
mean: 0.83527, std: 0.00826, params: {'n_neighbors': 158},
mean: 0.83554, std: 0.00832, params: {'n_neighbors': 159},
mean: 0.83554, std: 0.00778, params: {'n_neighbors': 160},
mean: 0.83519, std: 0.00745, params: {'n_neighbors': 161},
mean: 0.83519, std: 0.00757, params: {'n_neighbors': 162},
mean: 0.83536, std: 0.00715, params: {'n_neighbors': 163},
mean: 0.83527, std: 0.00767, params: {'n_neighbors': 164},
mean: 0.83536, std: 0.00778, params: {'n_neighbors': 165},
mean: 0.83492, std: 0.00759, params: {'n_neighbors': 166},
mean: 0.83510, std: 0.00790, params: {'n_neighbors': 167},
mean: 0.83527, std: 0.00820, params: {'n_neighbors': 168},
mean: 0.83492, std: 0.00775, params: {'n_neighbors': 169},
mean: 0.83456, std: 0.00746, params: {'n_neighbors': 170},
mean: 0.83447, std: 0.00782, params: {'n_neighbors': 171},
mean: 0.83376, std: 0.00770, params: {'n_neighbors': 172},
mean: 0.83376, std: 0.00731, params: {'n_neighbors': 173},
mean: 0.83270, std: 0.00737, params: {'n_neighbors': 174},
mean: 0.83287, std: 0.00736, params: {'n_neighbors': 175},
mean: 0.83261, std: 0.00701, params: {'n_neighbors': 176},
mean: 0.83234, std: 0.00676, params: {'n_neighbors': 177},
mean: 0.83225, std: 0.00719, params: {'n_neighbors': 178},
mean: 0.83207, std: 0.00718, params: {'n_neighbors': 179},
mean: 0.83181, std: 0.00746, params: {'n_neighbors': 180},
mean: 0.83190, std: 0.00741, params: {'n_neighbors': 181},
mean: 0.83243, std: 0.00808, params: {'n_neighbors': 182},
mean: 0.83261, std: 0.00827, params: {'n_neighbors': 183},
mean: 0.83225, std: 0.00819, params: {'n_neighbors': 184},
mean: 0.83279, std: 0.00779, params: {'n_neighbors': 185},
mean: 0.83118, std: 0.00688, params: {'n_neighbors': 186},
mean: 0.83154, std: 0.00684, params: {'n_neighbors': 187},
mean: 0.82950, std: 0.00716, params: {'n_neighbors': 188},
mean: 0.83003, std: 0.00689, params: {'n_neighbors': 189},
mean: 0.82870, std: 0.00649, params: {'n_neighbors': 190},
mean: 0.82878, std: 0.00704, params: {'n_neighbors': 191},
mean: 0.82665, std: 0.00624, params: {'n_neighbors': 192},
mean: 0.82718, std: 0.00654, params: {'n_neighbors': 193},
mean: 0.82478, std: 0.00895, params: {'n_neighbors': 194},
mean: 0.82550, std: 0.00846, params: {'n_neighbors': 195},
mean: 0.82230, std: 0.00974, params: {'n_neighbors': 196},
mean: 0.82274, std: 0.01041, params: {'n_neighbors': 197},
mean: 0.81936, std: 0.00926, params: {'n_neighbors': 198},
mean: 0.81972, std: 0.00963, params: {'n_neighbors': 199}]
In [104]:
# create a list of the mean scores only
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print(grid_mean_scores)
[0.9623966574806649, 0.9588407858476309, 0.9456840608054049, 0.94621744155036, 0.9387501111209885, 0.9428393634989777, 0.9379500400035559, 0.9383945239576851, 0.9335052004622633, 0.9349275491154769, 0.9304827095741843, 0.9325273357631789, 0.9293270512934483, 0.9296826384567517, 0.9271935283136279, 0.9281713930127122, 0.9255933860787625, 0.9272824251044537, 0.9249711085429816, 0.9247044181705041, 0.9220375144457285, 0.9218597208640769, 0.9190150235576495, 0.9177704684860877, 0.9156369455062672, 0.914570184016357, 0.9124366610365365, 0.9124366610365365, 0.911014312383323, 0.9111032091741488, 0.9090585829851542, 0.9094141701484576, 0.9077251311227664, 0.9072806471686372, 0.9055027113521202, 0.9041692594897324, 0.9029247044181705, 0.9009689750200017, 0.9003466974842208, 0.8987465552493555, 0.8975908969686194, 0.8962574451062316, 0.8953684771979732, 0.8927015734731976, 0.8917237087741132, 0.8895012890034669, 0.8890568050493377, 0.8866565916970397, 0.8857676237887813, 0.8852342430438261, 0.8845230687172193, 0.8837229975997867, 0.8828340296915281, 0.8814116810383145, 0.881144990665837, 0.8806116099208818, 0.879900435594275, 0.8791003644768424, 0.8789225708951907, 0.8780336029869322, 0.8776780158236288, 0.8768779447061961, 0.8765223575428926, 0.8759000800071117, 0.8755444928438083, 0.8747444217263757, 0.8741221441905948, 0.873944350608943, 0.8734109698639879, 0.8731442794915104, 0.8726997955373811, 0.8722553115832519, 0.871544137256645, 0.8707440661392124, 0.8705662725575607, 0.8700328918126056, 0.8696773046493022, 0.8675437816694818, 0.8669215041337007, 0.8656769490621389, 0.8659436394346164, 0.8646101875722286, 0.8638990132456218, 0.8623877678015823, 0.862565561383234, 0.8608765223575429, 0.8608765223575429, 0.8598097608676327, 0.8593652769135034, 0.8582985154235933, 0.8582096186327673, 0.8561649924437728, 0.8565205796070762, 0.8551871277446884, 0.8554538181171659, 0.853675882300649, 0.8533202951373455, 0.8519868432749578, 0.851897946484132, 0.8502089074584408, 0.8503867010400925, 0.8483420748510979, 0.848253178060272, 0.8455862743354965, 0.8455862743354965, 0.8441639256822828, 0.8445195128455862, 0.8437194417281536, 0.8439861321006311, 0.8425637834474176, 0.8423859898657658, 0.8413192283758556, 0.8408747444217264, 0.8403413636767713, 0.8401635700951195, 0.8398079829318161, 0.8397190861409903, 0.8386523246510801, 0.8389190150235577, 0.8388301182327318, 0.8389190150235577, 0.8366076984620855, 0.8370521824162148, 0.8366965952529114, 0.8371410792070406, 0.8373188727886923, 0.8374077695795181, 0.8373188727886923, 0.8374077695795181, 0.8375855631611698, 0.837496666370344, 0.8375855631611698, 0.8376744599519957, 0.8373188727886923, 0.8377633567428215, 0.8372299759978665, 0.8379411503244732, 0.8368743888345631, 0.8368743888345631, 0.8360743177171304, 0.8362521112987821, 0.8350075562272202, 0.8353631433905236, 0.8342963819006134, 0.8353631433905236, 0.835096453018046, 0.8352742465996977, 0.8341185883189617, 0.8349186594363943, 0.8347408658547426, 0.8348297626455685, 0.8349186594363943, 0.8353631433905236, 0.8349186594363943, 0.8349186594363943, 0.8349186594363943, 0.835096453018046, 0.8352742465996977, 0.8355409369721754, 0.8355409369721754, 0.8351853498088719, 0.8351853498088719, 0.8353631433905236, 0.8352742465996977, 0.8353631433905236, 0.8349186594363943, 0.835096453018046, 0.8352742465996977, 0.8349186594363943, 0.8345630722730909, 0.8344741754822651, 0.8337630011556583, 0.8337630011556583, 0.832696239665748, 0.8328740332473997, 0.8326073428749222, 0.8323406525024447, 0.8322517557116188, 0.8320739621299671, 0.8318072717574896, 0.8318961685483154, 0.8324295492932705, 0.8326073428749222, 0.8322517557116188, 0.8327851364565739, 0.8311849942217086, 0.831540581385012, 0.8294959551960174, 0.8300293359409725, 0.8286958840785847, 0.8287847808694107, 0.8266512578895902, 0.8271846386345453, 0.8247844252822473, 0.8254955996088541, 0.8222953151391235, 0.8227397990932528, 0.8193617210418704, 0.8197173082051737]
In [105]:
# plot the results (Manhattan distance)
plt.plot(k_potentials, grid_mean_scores)
plt.xlabel('Potential K values for KNN')
plt.ylabel('Cross-Validated Accuracy')
Out[105]:
<matplotlib.text.Text at 0x120cb5a50>

In [106]:
# plot the results (using Euclidean distance)
plt.plot(k_potentials, grid_mean_scores)
plt.xlabel('Potential K values for KNN')
plt.ylabel('Cross-Validated Accuracy')
Out[106]:
<matplotlib.text.Text at 0x12143b7d0>

In [107]:
# examine the best model
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
0.962396657481
{'n_neighbors': 1}
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
metric_params=None, n_jobs=1, n_neighbors=1, p=2,
weights='uniform')
In [ ]:

In [108]:
# read in the data & create matrices
X = X_scaled[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", \
"time_spend_company", "Work_accident", "promotion_last_5years", "low", "medium",\
"technical", "support", "IT", "product_mng", "marketing", "RandD", "accounting",\
"hr"]]
y = X_scaled["left"]
In [176]:
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
Support Vector Machines
In [177]:
# Train, fit and predict with SVM

from sklearn import svm
Model=svm.SVC(kernel='linear')
Model.fit(X_train,y_train)
Y_pred=Model.predict(X_test)
In [178]:
# Cross validation score for SVM
scores = cross_val_score(Model, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.780694712068
In [179]:
# Evaluate SVM for accuracy

from sklearn import metrics
metrics.accuracy_score(y_test,Y_pred)
Out[179]:
0.78106666666666669
In [180]:
# Confusion matrix for SVM

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2696  166]
[ 655  233]]
In [181]:
# Classification Accuracy confirmation between manual calculation and formula for SVM

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.781066666667
0.781066666667
In [182]:
# Misclassification Accuracy confirmation between manual calculation and formula for SVM
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.218933333333
0.218933333333
In [183]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated SVM

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.262387387387
0.262387387387
In [184]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for SVM

print(TN / float(TN + FP))
0.941998602376
In [185]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for SVM

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.583959899749
0.583959899749
In [186]:
# Misclassification Accuracy confirmation between manual calculation and formula for SVM
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.218933333333
0.218933333333
In [187]:
# Roc_curve for Support Vector Machine Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [92]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [93]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Support Vector Machines: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

Random Forest
In [157]:
# Random Forests Classifier from Scikit Learn

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_jobs=2)
RF.fit(X_train,y_train)
Y_pred=RF.predict(X_test)
In [158]:
# Score for 10-fold Cross Validation for Random Forests Classifier

scores = cross_val_score(RF, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.986309602501
In [159]:
# Prediction Accuracy for Random Forests

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.9896
In [160]:
# Confusion matrix for Random Forests

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2857    5]
[  34  854]]
In [161]:
# Classification Accuracy confirmation between manual calculation and formula for Random Forest

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.9896
0.9896
In [162]:
# Misclassification Accuracy confirmation between manual calculation and formula for Random Forest
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0104
0.0104
In [163]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for Random Forest Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.961711711712
0.961711711712
In [164]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for Random Forest Classifier

print(TN / float(TN + FP))
0.998252969951
In [165]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for Random Forest Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.994179278231
0.994179278231
In [166]:
# Misclassification Accuracy confirmation between manual calculation and formula for Random Forest
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0104
0.0104
In [167]:
# Roc_curve for Random Forest Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [168]:
y_pred_prob_Random_Forest = RF.predict_proba(X_test)[:, 1]
In [172]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_Random_Forest, bins=5)
plt.xlim(0, 1)
plt.title('Random Forest Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[172]:
<matplotlib.text.Text at 0x12013a8d0>

In [170]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [171]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Random Forest: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

Bagging
In [109]:
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
In [110]:
# Bagging Classifier

seed = 7
cart = DecisionTreeClassifier()
num_trees = 100
Bagging = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
Bagging.fit(X_train,y_train)
Y_pred=Bagging.predict(X_test)
In [111]:
# Cross Validation for Bagging

scores = cross_val_score(Bagging, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.987910076856
In [112]:
# Prediction Accuracy for Bagging

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.989333333333
In [113]:
# Confusion matrix for Bagging

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2854    8]
[  32  856]]
In [114]:
# Classification Accuracy confirmation between manual calculation and formula for Bagging

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.989333333333
0.989333333333
In [115]:
# Misclassification Accuracy confirmation between manual calculation and formula for Bagging
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0106666666667
0.0106666666667
In [116]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for Bagging Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.963963963964
0.963963963964
In [117]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for Bagging Classifier

print(TN / float(TN + FP))
0.997204751922
In [118]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for Bagging Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.990740740741
0.990740740741
In [119]:
# Misclassification Accuracy confirmation between manual calculation and formula for Bagging
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0106666666667
0.0106666666667
In [120]:
# Roc_curve for Bagging


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [121]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
In [122]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Bagging: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [123]:
y_pred_prob_Bagging = Bagging.predict_proba(X_test)[:, 1]
In [124]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_Bagging, bins=5)
plt.xlim(0, 1)
plt.title('Bagging Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[124]:
<matplotlib.text.Text at 0x11b217610>

In [ ]:

AdaBoost
In [125]:
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
In [126]:
# AdaBoost Classifier

seed = 7
cart = AdaBoostClassifier()
num_trees = 100
AdaBoostClassifier = AdaBoostClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
AdaBoostClassifier.fit(X_train,y_train)
Y_pred=AdaBoostClassifier.predict(X_test)
In [127]:
# Cross Validation for AdaBoost

scores = cross_val_score(AdaBoostClassifier, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.958576016468
In [128]:
# Prediction Accuracy for AdaBoost

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.958133333333
In [129]:
# Confusion matrix for AdaBoost

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2793   69]
[  88  800]]
In [130]:
# Classification Accuracy confirmation between manual calculation and formula for Bagging

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.958133333333
0.958133333333
In [131]:
# Misclassification Accuracy confirmation between manual calculation and formula for AdaBoost
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0418666666667
0.0418666666667
In [132]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for AdaBoost Boosting Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.900900900901
0.900900900901
In [133]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for AdaBoost Classifier

print(TN / float(TN + FP))
0.975890985325
In [134]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for AdaBoost Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.920598388953
0.920598388953
In [135]:
# Misclassification Accuracy confirmation between manual calculation and formula for AdaBoost
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0418666666667
0.0418666666667
In [136]:
# Roc_curve for AdaBoost


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [137]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
In [138]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='AdaBoost: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [139]:
y_pred_prob_AdaBoost = AdaBoostClassifier.predict_proba(X_test)[:, 1]
In [140]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_AdaBoost, bins=5)
plt.xlim(0, 1)
plt.title('AdaBoost Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[140]:
<matplotlib.text.Text at 0x11b367a10>

In [ ]:

In [ ]:

In [ ]:

Gradient Boosting
In [141]:
from sklearn.ensemble import GradientBoostingClassifier
In [142]:
# Train and predict Gradient Boosting

GradientBoosting = GradientBoostingClassifier()
GradientBoosting.fit(X_train,y_train)
Y_pred=GradientBoosting.predict(X_test)
In [143]:
# Cross Validation for Gradient Boosting

scores = cross_val_score(GradientBoosting, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.976443246546
In [144]:
# Prediction Accuracy for Gradient Boosting

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.974133333333
In [145]:
# Confusion matrix for Gradient Boosting

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2832   30]
[  67  821]]
In [146]:
# Classification Accuracy confirmation between manual calculation and formula for Gradient Boosting

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.974133333333
0.974133333333
In [147]:
# Misclassification Accuracy confirmation between manual calculation and formula for Gradient Boosting
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0258666666667
0.0258666666667
In [148]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for Gradient Boosting Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.92454954955
0.92454954955
In [149]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for Gradient Classifier

print(TN / float(TN + FP))
0.989517819706
In [150]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for Gradient Boosting Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.964747356052
0.964747356052
In [151]:
# Misclassification Accuracy confirmation between manual calculation and formula for Gradient Boosting
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0258666666667
0.0258666666667
In [152]:
# Roc_curve for Gradient Boosting


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Gradient Boosting Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [153]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
In [154]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Gradient Boosting: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [155]:
y_pred_prob_GradientBoosting = GradientBoosting.predict_proba(X_test)[:, 1]
In [156]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_GradientBoosting, bins=5)
plt.xlim(0, 1)
plt.title('Gradient Boosting Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[156]:
<matplotlib.text.Text at 0x11c2566d0>

In [ ]:

In [ ]:

Part IV
Conclusion
One interesting chart showed us which departments have the most people leaving (and staying). They are Sales followed by Technical and Support. That means that the biggest effort at the company should be placed on keeping the strong employees in those departments (but by no means neglect the other departments).
There are several other interesting patterns in the data. Some are very intuitive whereas others very surprising and unexpected. There are three clusters of people leaving the company (the most important cluster is not as dense, in the sense that there are actually slightly more people there that stay than leave, but nevertheless its the most interesting and surprising cluster).
The three groups are employees with high performance and low satisfaction, employees with low satisfcation and performance (rather obvious) and employees with high satisfaction and performance (a surprising result). The first two results are very strong but rather trivial, there is limited use in providing detailed analyses of them. However, the somewhat less dense but very significant group of highly satisfied employees that do remarkably well in their evaluations is of extreme importance. Why do they leave? Well, there are several important factors to consider.
The people with high salaries and at even average satisfaction levels rarely leave the firm. Hence, one attribute that people who tend to leave will generally possess is salaries that are not high and fall in the low or medium range. It is interesting to note though that people with near perfect to perfect satisfaction levels do not leave even if they are in the low or medium income range brackets. Hence, a possible course of action to potentially alleviate this issue is increase salaries for good employees. Also, a further investigation into what might be the differentiating factor between people who are very satisfied (0.7-0.9 rate) vs super satisfied (0.9-1.0) can be done to see if it is possible to further increase the satisfaction level of those that are very satisfied but not perfectly so (maybe its possible to do this while at the same time saving money by not instituting as much pay increases if there is a way to get them satisfied in another manner, perhaps by working less hours for instance) would be a good idea.
Secondly, promotions are a huge factor. Looking at the plot of promotions during the past five years, it is very clear that people who are promoted stay and people who are not promoted leave. Of course, promotions and salary increases have some correlation because people tend to receive salary increases during promotions. Hence, it's important to promote employees if they do good work! That will help decrease the turnover rate.
The third important point is that employees who work a lot of hours that stay at the company for at least a medium amount of time tend to leave. Only the relatively small number of employees who work at the company for many years stay at the company regardless of working hours but very few make it there. Hence, the hours worked plays an important decision making role when employees think about whether to stay at the firm or seek other employment. Due to this finding it is important to see if any kind of rebalancing can be done so that part of the work can perhaps be shifted between employees who do not work as many hours and those that do in order to make the distribution of hours worked by employees more uniform which can perhaps increase employees willingness to stay at the company longer. It is quiet possible that some employees just feel burnt out even if they are generally happy with their job. Due to an inefficient work life balance, they might seek to find employment elsewhere.
Multiple machine learning algorithms were ran on the data and quiet a few did relatively well in terms of making predictions on out of sample data as measured by the accuracy score, the confusion matrix and deriatives thereof as well as the ROC Curve. In particular, the Random Forest and Bagging algorithms did the best with accuracy and cross-validation scores marginally different from each other just under 99%. Both correctly predicted over 3,700 out of 3,750 employees accurately (compared to a baseline accuracy of only 76% (2,850) as to whether they would leave/stay). The various scores computed from the confusion matrix were all almost perfect. Particularly importantly, the ROC curve has an area covering of .98 for both those algorithms.
Hence, using either of these two algorithms (Random Forests or Bagging) have given us great predictions as to which employees will leave vs which ones would stay.
Assumptions
We have assumed the dataset is accurate as we have not audited its validity.
Another assumption we made is that we trained on 75% of the data, conducted cross-validation, and then tested on the 25% remaining out of sample. However, when trying to sample to train a larger sample of the data (such as 90/10 test/train split), we received very similar results in our algorithms. Hence, our model is robust as should perform well when used on future data.
An assumption made by the Random Forest Model which did exceptionally well is that the sampling that it conducted during boostrap aggregation is actually representative. This is a safe assumption to make and the model has done really well on out of sample data.
Introduction:
This project analyzes a dataset in an attempt to determine what causes the good employees to leave their current jobs. The first step is to clean the data (Part I) and visualize it (Part II) using various types of graphs and charts to help establish clear relationships between various features. Part III analyzes the dataset using various Machine Learning algorithms and picks the best algorithm to model this dataset based on various metrics. Lastly, a conclusion on the entire analysis is provided.
Part I
Data Cleaning
This initial stage of the project is to clean the data in such a way that all the important attributes of the dataset be retained and unhindered for analysis purposes. Hence, the initial cleaning phase of the project imports the dataset and the libraries needed for the cleaning stages of the project. A check for Null Values/Missing Values is conducted (none are present). The multiple categorical variables in the dataset are created into dummy variables which is necessary for later stages of the analysis. These new variables are afterwards added to the original dataset and the initial variables from which dummification was done are removed to avoid redundancy. Lastly, the dataset is normalized due to the need to apply machine learning algorithms to it in later stages of the project.
In [1]:
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

%matplotlib inline
In [2]:
HR = pd.read_csv('HR_comma_sep.csv')
HR.head(10)
Out[2]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary
0	0.38	0.53	2	157	3	0	1	0	sales	low
1	0.80	0.86	5	262	6	0	1	0	sales	medium
2	0.11	0.88	7	272	4	0	1	0	sales	medium
3	0.72	0.87	5	223	5	0	1	0	sales	low
4	0.37	0.52	2	159	3	0	1	0	sales	low
5	0.41	0.50	2	153	3	0	1	0	sales	low
6	0.10	0.77	6	247	4	0	1	0	sales	low
7	0.92	0.85	5	259	5	0	1	0	sales	low
8	0.89	1.00	5	224	5	0	1	0	sales	low
9	0.42	0.53	2	142	3	0	1	0	sales	low
In [3]:
HR[HR['satisfaction_level'] == np.nan]
HR[HR['last_evaluation'] == np.nan]
HR[HR['number_project'] == np.nan]
HR[HR['average_montly_hours'] == np.nan]
HR[HR['time_spend_company'] == np.nan]
HR[HR['Work_accident'] == np.nan]
HR[HR['left'] == np.nan]
HR[HR['sales'] == np.nan]
HR[HR['salary'] == np.nan]
Out[3]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary
In [4]:
# Create a heatmap, to check for null values.

plt.figure(figsize=(10,7))
sns.heatmap(HR.isnull(),yticklabels=False, cbar=False, cmap='viridis')
Out[4]:
<matplotlib.axes._subplots.AxesSubplot at 0x114c6ee50>

In [5]:
HR['sales'].value_counts()
Out[5]:
sales          4140
technical      2720
support        2229
IT             1227
product_mng     902
marketing       858
RandD           787
accounting      767
hr              739
management      630
Name: sales, dtype: int64
In [6]:
HR['salary'].value_counts()
Out[6]:
low       7316
medium    6446
high      1237
Name: salary, dtype: int64
In [7]:
HR['left'].value_counts()
Out[7]:
0    11428
1     3571
Name: left, dtype: int64
In [8]:
# Create dummy variables for all the departments

sales = pd.get_dummies(HR.sales)

sales.columns = ["IT", "RandD", "accounting", "hr", "management", "marketing", "product_mng", "sales_department", "support", "technical"]

sales.head()
Out[8]:
IT	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical
0	0	0	0	0	0	0	0	1	0	0
1	0	0	0	0	0	0	0	1	0	0
2	0	0	0	0	0	0	0	1	0	0
3	0	0	0	0	0	0	0	1	0	0
4	0	0	0	0	0	0	0	1	0	0
In [9]:
# Create dummy variables for the salary

salary = pd.get_dummies(HR.salary)

salary.columns = ['low', 'medium', 'high']

salary.head()
Out[9]:
low	medium	high
0	0	1	0
1	0	0	1
2	0	0	1
3	0	1	0
4	0	1	0
In [10]:
# Create dummy variable for stayed/left

left = pd.get_dummies(HR.left)

left.columns = ['left', 'stayed']

left.head(10)
Out[10]:
left	stayed
0	0	1
1	0	1
2	0	1
3	0	1
4	0	1
5	0	1
6	0	1
7	0	1
8	0	1
9	0	1
In [11]:
# Concatenate the original dataset with the newly created salary and sales dummified variables

HR = pd.concat([HR, salary, sales], axis = 1)

HR.head(10)
Out[11]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary	...	IT	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical
0	0.38	0.53	2	157	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
1	0.80	0.86	5	262	6	0	1	0	sales	medium	...	0	0	0	0	0	0	0	1	0	0
2	0.11	0.88	7	272	4	0	1	0	sales	medium	...	0	0	0	0	0	0	0	1	0	0
3	0.72	0.87	5	223	5	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
4	0.37	0.52	2	159	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
5	0.41	0.50	2	153	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
6	0.10	0.77	6	247	4	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
7	0.92	0.85	5	259	5	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
8	0.89	1.00	5	224	5	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
9	0.42	0.53	2	142	3	0	1	0	sales	low	...	0	0	0	0	0	0	0	1	0	0
10 rows × 23 columns
In [12]:
HR.dtypes
Out[12]:
satisfaction_level       float64
last_evaluation          float64
number_project             int64
average_montly_hours       int64
time_spend_company         int64
Work_accident              int64
left                       int64
promotion_last_5years      int64
sales                     object
salary                    object
low                        uint8
medium                     uint8
high                       uint8
IT                         uint8
RandD                      uint8
accounting                 uint8
hr                         uint8
management                 uint8
marketing                  uint8
product_mng                uint8
sales_department           uint8
support                    uint8
technical                  uint8
dtype: object
In [13]:
# Create a new column 

salary_map = {'low': 1, 'medium': 2, 'high': 3}

HR['salary_variable'] = HR['salary'].apply(lambda x: salary_map[x])

HR.head()
Out[13]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	sales	salary	...	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical	salary_variable
0	0.38	0.53	2	157	3	0	1	0	sales	low	...	0	0	0	0	0	0	1	0	0	1
1	0.80	0.86	5	262	6	0	1	0	sales	medium	...	0	0	0	0	0	0	1	0	0	2
2	0.11	0.88	7	272	4	0	1	0	sales	medium	...	0	0	0	0	0	0	1	0	0	2
3	0.72	0.87	5	223	5	0	1	0	sales	low	...	0	0	0	0	0	0	1	0	0	1
4	0.37	0.52	2	159	3	0	1	0	sales	low	...	0	0	0	0	0	0	1	0	0	1
5 rows × 24 columns
In [14]:
del HR['salary']

del HR['sales']
In [15]:
HR.head()
Out[15]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	low	medium	...	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical	salary_variable
0	0.38	0.53	2	157	3	0	1	0	0	1	...	0	0	0	0	0	0	1	0	0	1
1	0.80	0.86	5	262	6	0	1	0	0	0	...	0	0	0	0	0	0	1	0	0	2
2	0.11	0.88	7	272	4	0	1	0	0	0	...	0	0	0	0	0	0	1	0	0	2
3	0.72	0.87	5	223	5	0	1	0	0	1	...	0	0	0	0	0	0	1	0	0	1
4	0.37	0.52	2	159	3	0	1	0	0	1	...	0	0	0	0	0	0	1	0	0	1
5 rows × 22 columns
In [16]:
HR.dtypes
Out[16]:
satisfaction_level       float64
last_evaluation          float64
number_project             int64
average_montly_hours       int64
time_spend_company         int64
Work_accident              int64
left                       int64
promotion_last_5years      int64
low                        uint8
medium                     uint8
high                       uint8
IT                         uint8
RandD                      uint8
accounting                 uint8
hr                         uint8
management                 uint8
marketing                  uint8
product_mng                uint8
sales_department           uint8
support                    uint8
technical                  uint8
salary_variable            int64
dtype: object
In [17]:
from sklearn.preprocessing import MinMaxScaler
X_scaled = MinMaxScaler().fit_transform(HR.astype(float))
X_scaled = pd.DataFrame(X_scaled, columns = HR.columns)

# Min Max Scaler

X_scaled.head(10)
Out[17]:
satisfaction_level	last_evaluation	number_project	average_montly_hours	time_spend_company	Work_accident	left	promotion_last_5years	low	medium	...	RandD	accounting	hr	management	marketing	product_mng	sales_department	support	technical	salary_variable
0	0.318681	0.265625	0.0	0.285047	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
1	0.780220	0.781250	0.6	0.775701	0.500	0.0	1.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.5
2	0.021978	0.812500	1.0	0.822430	0.250	0.0	1.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.5
3	0.692308	0.796875	0.6	0.593458	0.375	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
4	0.307692	0.250000	0.0	0.294393	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
5	0.351648	0.218750	0.0	0.266355	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
6	0.010989	0.640625	0.8	0.705607	0.250	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
7	0.912088	0.765625	0.6	0.761682	0.375	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
8	0.879121	1.000000	0.6	0.598131	0.375	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
9	0.362637	0.265625	0.0	0.214953	0.125	0.0	1.0	0.0	0.0	1.0	...	0.0	0.0	0.0	0.0	0.0	0.0	1.0	0.0	0.0	0.0
10 rows × 22 columns
Part II
Exploratory Data Analysis (EDA)
In this section, we try to visualize the data on employees using various plots, graphs, charts in order to get an idea of how different features in the dataset relate to each other.
The below represents a heatmap of correlations of features. Some interesting observations from this heatmap: Quiet intuitively, you can see that the relationship between satisfaction level and people that left the company is negative. It's also interesting to note that there is a positive correlation between the number of hours employees work and their evaluation. These and other relationships will be explored more deeply in the rest of the EDA.
In [18]:
correlation = X_scaled.corr()
plt.figure(figsize=(20,20))
sns.heatmap(correlation, vmax=10, square=True,annot=True,cmap=None, linewidths=1)

plt.title('Correlation between features')
Out[18]:
<matplotlib.text.Text at 0x118f3a050>

In [19]:
# Exported csv file to work with in Tableau
X_scaled.to_csv("Updated_Capstone.csv")
In [20]:
X_scaled['left'].value_counts()
Out[20]:
0.0    11428
1.0     3571
Name: left, dtype: int64
In [21]:
stayed = np.where(X_scaled['left'] == 0)[0]
left = np.where(X_scaled['left'] == 1)[0]
For a given level of satisfaction, there is definitely an intuitive pattern that the higher someone's salary is, the less they are likely to leave. This relationship is a lot more apparent in the distinction between the high income earners vs low & median income earners as a group. The difference in the people that left between low and median income salaries is not significant. Hence, most of the people that leave are not receiving a high salary, which means that you if you would like to retain these employees, you should increase their pay!
In [22]:
# plot of Satisfaction vs Last Evaluation
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.satisfaction_level[stayed], X_scaled.salary_variable[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.satisfaction_level[left], X_scaled.salary_variable[left], alpha=0.1, color='r')
ax.set_title("satisfaction_level vs salary")
ax.set_xlabel("satisfaction_level")
ax.set_ylabel("salary")
plt.show()

In the Satisfaction vs Last Evaluation plot below, we can note some interesting observations. Firstly, there is a sizeable cluster of people with strong evaluations and low satisfaction levels leaving. Secondly, there is a significant group of people leaving that are slighly below average in performance and evaluation. Lastly, and by far the most interesting observation of the three, there is a decent size of a not very dense cluster of people leaving that have done extremely well on their evaluations (0.7 to 1 on a 0-1 scale) and and have very high satisfaction rates, albeit not perfect (0.7-0.9 on a 0-1 scale). That particular group of employees also have many people that stay, but a significant proportion of that group is leaving. So the most surprising thing to observe is that people can be satisfied with the job and still leave! There are several possibilites for why this might be possible. People might enjoy their job but think they can do even better elsewhere, whether its income or career growth (or a combination of the two, considering that those things tend to go hand in hand). Another possibility is that people might not be honest in surveys because of fear of being reprimanded for negative surveys.
In [23]:
# plot of Satisfaction vs Last Evaluation
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.satisfaction_level[stayed], X_scaled.last_evaluation[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.satisfaction_level[left], X_scaled.last_evaluation[left], alpha=0.1, color='r')
ax.set_title("satisfaction_level vs last_evaluation")
ax.set_xlabel("satisfaction_level")
ax.set_ylabel("last_evaluation")
plt.show()

The Satisfaction vs Promotion during the last five years plot below shows a very strong relationship in several interesting ways! People who have been promoted don't leave very often! However, such promotions are not common enough! Therefore, if you don't want good employees to leave, promoting them will really help!
In [24]:
# plot of Satisfaction vs Promotion during last five years
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.satisfaction_level[stayed], X_scaled.promotion_last_5years[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.satisfaction_level[left], X_scaled.promotion_last_5years[left], alpha=0.1, color='r')
ax.set_title("satisfaction_level vs promotion_last_5years")
ax.set_xlabel("satisfaction_level")
ax.set_ylabel("promotion_last_5years")
plt.show()

The below scatter plot compares the average working hours against the time spent at the company and how it impacts the employee turnover rate. There are multiple interesting relationships to note. For employees that just started out, the majority tend to stick around for some time even if they work substantial hours. However, for employees who have been at the company for a while, there is a significant tendency to leave the company if working substantial hours. Consequently, we can observe that very few employees stick around at the company for a long time period because the ones who were working long hours almost all left! However, those that do stick around for a long time (relatively few) tend to all be devoted and not leave
There are several conjectures for why this is possible. One is that employees have worked for a significant amount of hours and put in a lot of time and effort expecting a quick promotion and/or pay increase (refer to graph above for relationship between promotions and staying/leaving) that they have not received and leave for elsewhere. Another reason might be that people have tried working for a little bit of time, realized that they are being overworked and have started looking for other opportunities to seek employment.
In [25]:
# plot of Average_monthly_hours vs time_spend_company
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.average_montly_hours[stayed], X_scaled.time_spend_company[stayed], alpha=0.1, color='g')
ax.scatter(X_scaled.average_montly_hours[left], X_scaled.time_spend_company[left], alpha=0.1, color='r')
ax.set_title("average_monthly_hours vs time_spend_company")
ax.set_xlabel("average_monthly_hours")
ax.set_ylabel("time_spend_company")
plt.show()

In [26]:
# Imported from Tableau
from IPython.display import Image
Image(filename = 'Average Monthly Hours vs Satisfaction Level.png', width = 1000)
Out[26]:

There does not appear to be a strong relationship between accidents and people leaving the company, hence, people stay/leave regardless of the accidents they might have experienced.
In [27]:
# plot of work_accident vs time_spend_company
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(X_scaled.Work_accident[stayed], X_scaled.time_spend_company[stayed], alpha=0.5, color='g')
ax.scatter(X_scaled.Work_accident[left], X_scaled.time_spend_company[left], alpha=0.5, color='r')
ax.set_title("Work_accident vs time_spend_company")
ax.set_xlabel("Work_accident")
ax.set_ylabel("time_spend_company")
plt.show()

The below stripplot compares number of projects done by an employee versus the number of promotions they have received over the past 5 years. It is interesting to note that the number of projects completed has very insignificant impact on whether a person leaves. On the contrary, the promotions received during the past 5 years is of critical importance. The majority of people who have received a promotion stay. Almost all those who were not promoted leave.
The data seems insufficient to be able to determine why the number of projects is not related to promotions. There are at least several possibilities. Perhaps the number of projects refers to the number assigned (vs completed) or that the projects are not all of equal difficulty and not equally time consuming leading to distorted hypothesis that the number of projects might necessarily correspond to an employee's productivity.
In [28]:
import seaborn as sns

sns.stripplot("number_project", "promotion_last_5years", data=X_scaled, hue="left", jitter = True)
Out[28]:
<matplotlib.axes._subplots.AxesSubplot at 0x117949d90>

The Satisfaction level distribution below shows that the majority of people are at least moderately satisfied with their job. Therefore, for the people that are satisfied and do well, efforts should be placed on making them even more happy by promoting them. For the people that do poorly or mediocre, either studies should be placed on determining why they are not doing well and trying to improve their performance or attempting to replace them with better employees (this would depend on a cost/benefit analysis) done by the company regarding this issue.
In [29]:
# Histogram distribution of Satisfaction Level

f,ax1 = plt.subplots(1,1)
sns.distplot(X_scaled['satisfaction_level'],bins=10,norm_hist=False)
plt.show()

The distribution (which is a decent approximation of a normal distribution with almost no tails and some kurtosis present) of the last evaluation graph shows us there is a sizeable portion of people that do well in their evaluations, hence the company should try to do as much as possible to keep those people!
In [30]:
# Histogram distribution of Last Evaluation


f,ax1 = plt.subplots(1,1)
sns.distplot(X_scaled['last_evaluation'],bins=10,norm_hist=False)
plt.show()

The below plot shows the ratios of people who stay in the company by department and the ratio of people who leave the company by department. It is clear from the chart that the departments with the most number of employees also have the highest ratios of both people who have left and those that stayed at the company. This tells us there should be particular effort place on retaining good employees that work in those departments, in particular the sales, technical and support departments, respectively. The largest efforts should be placed on retaining the good employees in these groups because they have the most people leaving.
In [34]:
# This plot is to compare the people that left across different departments
# The non-department variables are deleted to only keep the departments

del X_scaled['satisfaction_level']
del X_scaled['last_evaluation']
del X_scaled['number_project']
del X_scaled['average_montly_hours']
del X_scaled['time_spend_company']
del X_scaled['Work_accident']
del X_scaled['promotion_last_5years']
del X_scaled['low']
del X_scaled['medium']
del X_scaled['high']
del X_scaled['salary_variable']


left=X_scaled.groupby('left').mean()
left=left.transpose()
plt.figure(figsize=(20,10))
left.plot(kind='bar')
Out[34]:
<matplotlib.axes._subplots.AxesSubplot at 0x11a7bcd10>
<matplotlib.figure.Figure at 0x11a360850>

Part III
Predictive Modelling
In this section below, multiple models will be run that are applicable to classification. Afterwards, various scoring metrics will be calculated for the models and a determination will be made as to which model is based for this dataset.
In [174]:
# read in the data & create matrices
X = X_scaled[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", \
"time_spend_company", "Work_accident", "promotion_last_5years", "salary_variable",\
"technical", "support", "IT", "product_mng", "marketing", "RandD", "accounting",\
"hr", "management", "sales_department"]]
y = X_scaled["left"]
In [175]:
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
Logistic Regression Model
In [33]:
# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
model = LogisticRegression()

# fit the model with data
mdl = model.fit(X_train, y_train)

# predict the response for new observations

logit = model.predict(X_test)

# sklearn output to check that ran on right data split.

len(logit)
Out[33]:
3750
In [34]:
# These represent the y-intercept and coefficients for all the variables in the logistic regression.

print(model.intercept_)
print(model.coef_)
[ 0.57305398]
[[-3.71487625  0.45484967 -1.50619896  0.8704181   1.97590531 -1.45425383
-1.26024249 -1.32512177  0.22588571  0.23497924 -0.01282827  0.07509077
0.13925062 -0.47542127  0.26606552  0.3920348  -0.44451555  0.17251241]]
In [35]:
# Prediction Accuracy for Logistic Regression

from sklearn import metrics
print(metrics.accuracy_score(y_test, logit))
0.796
In [36]:
from sklearn.cross_validation import cross_val_score
In [37]:
# 10-fold cross-validation for Logistic Regression to find the CV score
mdl = model.fit(X_train, y_train)
scores = cross_val_score(mdl, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.794205514996
In [38]:
from sklearn import metrics
confusion = metrics.confusion_matrix(y_test, logit)
print ((confusion))
[[2651  211]
[ 554  334]]
In [39]:
y_pred_prob = model.predict_proba(X_test)[:, 1]
In [40]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob, bins=5)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[40]:
<matplotlib.text.Text at 0x1181a6a90>

In [41]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [42]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, logit)


plot_confusion_matrix(cm, title='LogisticRegression: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [43]:
# Roc_curve for Logistic Regression Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, logit)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Logistic Regression Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

K Nearest Nearbours Model
K=1
In [44]:
from sklearn.neighbors import KNeighborsClassifier
In [45]:
# K-Nearest Neighbours with K=1
knn = KNeighborsClassifier(n_neighbors=1, metric = 'euclidean')
In [46]:
y = knn.fit(X_train, y_train)

KNN = knn.predict(X_test)

# sklearn output to check that ran on right data split.

len(KNN)
Out[46]:
3750
In [47]:
# Cross fold validation for K=1

md2 = knn.fit(X_train, y_train)
scores = cross_val_score(md2, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.962043001151
In [48]:
# Prediction Accuracy.

from sklearn import metrics
print(metrics.accuracy_score(y_test, KNN))
0.967466666667
In [49]:
confusion = metrics.confusion_matrix(y_test, KNN)
print confusion
[[2775   87]
[  35  853]]
In [50]:
# Classification Accuracy confirmation between manual calculation and formula for K=1

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, KNN))
0.967466666667
0.967466666667
In [51]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=1
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN))
0.0325333333333
0.0325333333333
In [52]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for K=1

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, KNN))
0.960585585586
0.960585585586
In [53]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for K=1


print(TN / float(TN + FP))
0.969601677149
In [54]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for K=1

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, KNN))
0.907446808511
0.907446808511
In [55]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=1
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN))
0.0325333333333
0.0325333333333
In [56]:
y_pred_prob_knn = knn.predict_proba(X_test)[:, 1]
In [57]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_knn, bins=5)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[57]:
<matplotlib.text.Text at 0x1181d6b90>

In [58]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [59]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, KNN)


plot_confusion_matrix(cm, title='KNN for K=1: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [60]:
# Roc_curve for KNN=1 Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, KNN)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('KNN=1 Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

K=5
In [61]:
# K-Nearest Neighbours with K=5

knn = KNeighborsClassifier(n_neighbors=5, metric = 'euclidean')
y = knn.fit(X_train, y_train)
KNN5 = knn.predict(X_test)

# sklearn output to check that ran on right data split.

len(KNN5)
Out[61]:
3750
In [62]:
# Cross Validation for K=5 Cross-Fold Validation

md2 = knn.fit(X_train, y_train)
scores = cross_val_score(md2, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.938309179837
In [63]:
# Prediction Accuracy on KNN=5 for accuracy score

from sklearn import metrics
print(metrics.accuracy_score(y_test, KNN5))
0.941333333333
In [64]:
confusion = metrics.confusion_matrix(y_test, KNN5)
print confusion
[[2731  131]
[  89  799]]
In [65]:
# Classification Accuracy confirmation between manual calculation and formula for K=5

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, KNN5))
0.941333333333
0.941333333333
In [66]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=5
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN5))
0.0586666666667
0.0586666666667
In [67]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for K=5

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, KNN5))
0.899774774775
0.899774774775
In [68]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for K=5

print(TN / float(TN + FP))
0.954227812718
In [69]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for K=5

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, KNN5))
0.859139784946
0.859139784946
In [70]:
# Misclassification Accuracy confirmation between manual calculation and formula for K=5
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, KNN5))
0.0586666666667
0.0586666666667
In [71]:
# Roc_curve for KNN=5 Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, KNN5)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [72]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [73]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, KNN5)


plot_confusion_matrix(cm, title='KNN for N=5: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [74]:
y_pred_prob_knn = knn.predict_proba(X_test)[:, 1]
In [75]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_knn, bins=5)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[75]:
<matplotlib.text.Text at 0x11acdea10>

In [ ]:

Grid Search for optimal N value in KNN
In [101]:
from sklearn.grid_search import GridSearchCV
/Users/Misha/anaconda/lib/python2.7/site-packages/sklearn/grid_search.py:43: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.
DeprecationWarning)
In [102]:
k_potentials = list(range(1, 200))
print(k_potentials)
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]
In [103]:
# Set Nearest Neighbours equal to k
knn = KNeighborsClassifier(n_neighbors=k_potentials, metric = 'euclidean')
# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_potentials)
# instantiate the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
# fit the grid with data
grid.fit(X_train, y_train)
# view the results
grid.grid_scores_
Out[103]:
[mean: 0.96240, std: 0.00591, params: {'n_neighbors': 1},
mean: 0.95884, std: 0.00543, params: {'n_neighbors': 2},
mean: 0.94568, std: 0.00463, params: {'n_neighbors': 3},
mean: 0.94622, std: 0.00555, params: {'n_neighbors': 4},
mean: 0.93875, std: 0.00620, params: {'n_neighbors': 5},
mean: 0.94284, std: 0.00673, params: {'n_neighbors': 6},
mean: 0.93795, std: 0.00746, params: {'n_neighbors': 7},
mean: 0.93839, std: 0.00819, params: {'n_neighbors': 8},
mean: 0.93351, std: 0.00932, params: {'n_neighbors': 9},
mean: 0.93493, std: 0.00848, params: {'n_neighbors': 10},
mean: 0.93048, std: 0.00825, params: {'n_neighbors': 11},
mean: 0.93253, std: 0.00810, params: {'n_neighbors': 12},
mean: 0.92933, std: 0.00898, params: {'n_neighbors': 13},
mean: 0.92968, std: 0.00824, params: {'n_neighbors': 14},
mean: 0.92719, std: 0.00747, params: {'n_neighbors': 15},
mean: 0.92817, std: 0.00830, params: {'n_neighbors': 16},
mean: 0.92559, std: 0.00873, params: {'n_neighbors': 17},
mean: 0.92728, std: 0.00821, params: {'n_neighbors': 18},
mean: 0.92497, std: 0.00821, params: {'n_neighbors': 19},
mean: 0.92470, std: 0.00830, params: {'n_neighbors': 20},
mean: 0.92204, std: 0.00804, params: {'n_neighbors': 21},
mean: 0.92186, std: 0.00867, params: {'n_neighbors': 22},
mean: 0.91902, std: 0.00856, params: {'n_neighbors': 23},
mean: 0.91777, std: 0.00850, params: {'n_neighbors': 24},
mean: 0.91564, std: 0.00809, params: {'n_neighbors': 25},
mean: 0.91457, std: 0.00913, params: {'n_neighbors': 26},
mean: 0.91244, std: 0.00927, params: {'n_neighbors': 27},
mean: 0.91244, std: 0.00873, params: {'n_neighbors': 28},
mean: 0.91101, std: 0.00878, params: {'n_neighbors': 29},
mean: 0.91110, std: 0.00853, params: {'n_neighbors': 30},
mean: 0.90906, std: 0.00855, params: {'n_neighbors': 31},
mean: 0.90941, std: 0.00906, params: {'n_neighbors': 32},
mean: 0.90773, std: 0.00922, params: {'n_neighbors': 33},
mean: 0.90728, std: 0.00991, params: {'n_neighbors': 34},
mean: 0.90550, std: 0.00946, params: {'n_neighbors': 35},
mean: 0.90417, std: 0.00891, params: {'n_neighbors': 36},
mean: 0.90292, std: 0.00887, params: {'n_neighbors': 37},
mean: 0.90097, std: 0.00881, params: {'n_neighbors': 38},
mean: 0.90035, std: 0.00948, params: {'n_neighbors': 39},
mean: 0.89875, std: 0.01009, params: {'n_neighbors': 40},
mean: 0.89759, std: 0.01009, params: {'n_neighbors': 41},
mean: 0.89626, std: 0.00955, params: {'n_neighbors': 42},
mean: 0.89537, std: 0.01016, params: {'n_neighbors': 43},
mean: 0.89270, std: 0.01221, params: {'n_neighbors': 44},
mean: 0.89172, std: 0.01244, params: {'n_neighbors': 45},
mean: 0.88950, std: 0.01089, params: {'n_neighbors': 46},
mean: 0.88906, std: 0.01150, params: {'n_neighbors': 47},
mean: 0.88666, std: 0.01137, params: {'n_neighbors': 48},
mean: 0.88577, std: 0.01143, params: {'n_neighbors': 49},
mean: 0.88523, std: 0.01132, params: {'n_neighbors': 50},
mean: 0.88452, std: 0.01149, params: {'n_neighbors': 51},
mean: 0.88372, std: 0.01237, params: {'n_neighbors': 52},
mean: 0.88283, std: 0.01227, params: {'n_neighbors': 53},
mean: 0.88141, std: 0.01166, params: {'n_neighbors': 54},
mean: 0.88114, std: 0.01085, params: {'n_neighbors': 55},
mean: 0.88061, std: 0.01041, params: {'n_neighbors': 56},
mean: 0.87990, std: 0.01105, params: {'n_neighbors': 57},
mean: 0.87910, std: 0.00974, params: {'n_neighbors': 58},
mean: 0.87892, std: 0.00997, params: {'n_neighbors': 59},
mean: 0.87803, std: 0.01108, params: {'n_neighbors': 60},
mean: 0.87768, std: 0.01119, params: {'n_neighbors': 61},
mean: 0.87688, std: 0.01152, params: {'n_neighbors': 62},
mean: 0.87652, std: 0.01182, params: {'n_neighbors': 63},
mean: 0.87590, std: 0.01129, params: {'n_neighbors': 64},
mean: 0.87554, std: 0.01148, params: {'n_neighbors': 65},
mean: 0.87474, std: 0.01183, params: {'n_neighbors': 66},
mean: 0.87412, std: 0.01223, params: {'n_neighbors': 67},
mean: 0.87394, std: 0.01196, params: {'n_neighbors': 68},
mean: 0.87341, std: 0.01210, params: {'n_neighbors': 69},
mean: 0.87314, std: 0.01231, params: {'n_neighbors': 70},
mean: 0.87270, std: 0.01219, params: {'n_neighbors': 71},
mean: 0.87226, std: 0.01168, params: {'n_neighbors': 72},
mean: 0.87154, std: 0.01185, params: {'n_neighbors': 73},
mean: 0.87074, std: 0.01104, params: {'n_neighbors': 74},
mean: 0.87057, std: 0.01132, params: {'n_neighbors': 75},
mean: 0.87003, std: 0.01129, params: {'n_neighbors': 76},
mean: 0.86968, std: 0.01148, params: {'n_neighbors': 77},
mean: 0.86754, std: 0.01056, params: {'n_neighbors': 78},
mean: 0.86692, std: 0.01041, params: {'n_neighbors': 79},
mean: 0.86568, std: 0.01028, params: {'n_neighbors': 80},
mean: 0.86594, std: 0.01024, params: {'n_neighbors': 81},
mean: 0.86461, std: 0.00937, params: {'n_neighbors': 82},
mean: 0.86390, std: 0.00915, params: {'n_neighbors': 83},
mean: 0.86239, std: 0.00956, params: {'n_neighbors': 84},
mean: 0.86257, std: 0.01001, params: {'n_neighbors': 85},
mean: 0.86088, std: 0.01023, params: {'n_neighbors': 86},
mean: 0.86088, std: 0.01051, params: {'n_neighbors': 87},
mean: 0.85981, std: 0.01214, params: {'n_neighbors': 88},
mean: 0.85937, std: 0.01208, params: {'n_neighbors': 89},
mean: 0.85830, std: 0.01087, params: {'n_neighbors': 90},
mean: 0.85821, std: 0.01108, params: {'n_neighbors': 91},
mean: 0.85616, std: 0.00995, params: {'n_neighbors': 92},
mean: 0.85652, std: 0.00954, params: {'n_neighbors': 93},
mean: 0.85519, std: 0.00892, params: {'n_neighbors': 94},
mean: 0.85545, std: 0.00957, params: {'n_neighbors': 95},
mean: 0.85368, std: 0.00900, params: {'n_neighbors': 96},
mean: 0.85332, std: 0.00845, params: {'n_neighbors': 97},
mean: 0.85199, std: 0.00923, params: {'n_neighbors': 98},
mean: 0.85190, std: 0.00926, params: {'n_neighbors': 99},
mean: 0.85021, std: 0.00825, params: {'n_neighbors': 100},
mean: 0.85039, std: 0.00930, params: {'n_neighbors': 101},
mean: 0.84834, std: 0.01015, params: {'n_neighbors': 102},
mean: 0.84825, std: 0.01016, params: {'n_neighbors': 103},
mean: 0.84559, std: 0.00954, params: {'n_neighbors': 104},
mean: 0.84559, std: 0.00922, params: {'n_neighbors': 105},
mean: 0.84416, std: 0.00977, params: {'n_neighbors': 106},
mean: 0.84452, std: 0.00978, params: {'n_neighbors': 107},
mean: 0.84372, std: 0.00935, params: {'n_neighbors': 108},
mean: 0.84399, std: 0.00922, params: {'n_neighbors': 109},
mean: 0.84256, std: 0.00943, params: {'n_neighbors': 110},
mean: 0.84239, std: 0.00930, params: {'n_neighbors': 111},
mean: 0.84132, std: 0.00941, params: {'n_neighbors': 112},
mean: 0.84087, std: 0.00924, params: {'n_neighbors': 113},
mean: 0.84034, std: 0.00864, params: {'n_neighbors': 114},
mean: 0.84016, std: 0.00875, params: {'n_neighbors': 115},
mean: 0.83981, std: 0.00849, params: {'n_neighbors': 116},
mean: 0.83972, std: 0.00862, params: {'n_neighbors': 117},
mean: 0.83865, std: 0.00779, params: {'n_neighbors': 118},
mean: 0.83892, std: 0.00782, params: {'n_neighbors': 119},
mean: 0.83883, std: 0.00798, params: {'n_neighbors': 120},
mean: 0.83892, std: 0.00810, params: {'n_neighbors': 121},
mean: 0.83661, std: 0.00577, params: {'n_neighbors': 122},
mean: 0.83705, std: 0.00582, params: {'n_neighbors': 123},
mean: 0.83670, std: 0.00582, params: {'n_neighbors': 124},
mean: 0.83714, std: 0.00545, params: {'n_neighbors': 125},
mean: 0.83732, std: 0.00565, params: {'n_neighbors': 126},
mean: 0.83741, std: 0.00610, params: {'n_neighbors': 127},
mean: 0.83732, std: 0.00647, params: {'n_neighbors': 128},
mean: 0.83741, std: 0.00634, params: {'n_neighbors': 129},
mean: 0.83759, std: 0.00626, params: {'n_neighbors': 130},
mean: 0.83750, std: 0.00609, params: {'n_neighbors': 131},
mean: 0.83759, std: 0.00639, params: {'n_neighbors': 132},
mean: 0.83767, std: 0.00616, params: {'n_neighbors': 133},
mean: 0.83732, std: 0.00640, params: {'n_neighbors': 134},
mean: 0.83776, std: 0.00641, params: {'n_neighbors': 135},
mean: 0.83723, std: 0.00690, params: {'n_neighbors': 136},
mean: 0.83794, std: 0.00675, params: {'n_neighbors': 137},
mean: 0.83687, std: 0.00710, params: {'n_neighbors': 138},
mean: 0.83687, std: 0.00718, params: {'n_neighbors': 139},
mean: 0.83607, std: 0.00744, params: {'n_neighbors': 140},
mean: 0.83625, std: 0.00725, params: {'n_neighbors': 141},
mean: 0.83501, std: 0.00739, params: {'n_neighbors': 142},
mean: 0.83536, std: 0.00761, params: {'n_neighbors': 143},
mean: 0.83430, std: 0.00774, params: {'n_neighbors': 144},
mean: 0.83536, std: 0.00776, params: {'n_neighbors': 145},
mean: 0.83510, std: 0.00746, params: {'n_neighbors': 146},
mean: 0.83527, std: 0.00732, params: {'n_neighbors': 147},
mean: 0.83412, std: 0.00716, params: {'n_neighbors': 148},
mean: 0.83492, std: 0.00704, params: {'n_neighbors': 149},
mean: 0.83474, std: 0.00740, params: {'n_neighbors': 150},
mean: 0.83483, std: 0.00759, params: {'n_neighbors': 151},
mean: 0.83492, std: 0.00751, params: {'n_neighbors': 152},
mean: 0.83536, std: 0.00788, params: {'n_neighbors': 153},
mean: 0.83492, std: 0.00833, params: {'n_neighbors': 154},
mean: 0.83492, std: 0.00841, params: {'n_neighbors': 155},
mean: 0.83492, std: 0.00815, params: {'n_neighbors': 156},
mean: 0.83510, std: 0.00822, params: {'n_neighbors': 157},
mean: 0.83527, std: 0.00826, params: {'n_neighbors': 158},
mean: 0.83554, std: 0.00832, params: {'n_neighbors': 159},
mean: 0.83554, std: 0.00778, params: {'n_neighbors': 160},
mean: 0.83519, std: 0.00745, params: {'n_neighbors': 161},
mean: 0.83519, std: 0.00757, params: {'n_neighbors': 162},
mean: 0.83536, std: 0.00715, params: {'n_neighbors': 163},
mean: 0.83527, std: 0.00767, params: {'n_neighbors': 164},
mean: 0.83536, std: 0.00778, params: {'n_neighbors': 165},
mean: 0.83492, std: 0.00759, params: {'n_neighbors': 166},
mean: 0.83510, std: 0.00790, params: {'n_neighbors': 167},
mean: 0.83527, std: 0.00820, params: {'n_neighbors': 168},
mean: 0.83492, std: 0.00775, params: {'n_neighbors': 169},
mean: 0.83456, std: 0.00746, params: {'n_neighbors': 170},
mean: 0.83447, std: 0.00782, params: {'n_neighbors': 171},
mean: 0.83376, std: 0.00770, params: {'n_neighbors': 172},
mean: 0.83376, std: 0.00731, params: {'n_neighbors': 173},
mean: 0.83270, std: 0.00737, params: {'n_neighbors': 174},
mean: 0.83287, std: 0.00736, params: {'n_neighbors': 175},
mean: 0.83261, std: 0.00701, params: {'n_neighbors': 176},
mean: 0.83234, std: 0.00676, params: {'n_neighbors': 177},
mean: 0.83225, std: 0.00719, params: {'n_neighbors': 178},
mean: 0.83207, std: 0.00718, params: {'n_neighbors': 179},
mean: 0.83181, std: 0.00746, params: {'n_neighbors': 180},
mean: 0.83190, std: 0.00741, params: {'n_neighbors': 181},
mean: 0.83243, std: 0.00808, params: {'n_neighbors': 182},
mean: 0.83261, std: 0.00827, params: {'n_neighbors': 183},
mean: 0.83225, std: 0.00819, params: {'n_neighbors': 184},
mean: 0.83279, std: 0.00779, params: {'n_neighbors': 185},
mean: 0.83118, std: 0.00688, params: {'n_neighbors': 186},
mean: 0.83154, std: 0.00684, params: {'n_neighbors': 187},
mean: 0.82950, std: 0.00716, params: {'n_neighbors': 188},
mean: 0.83003, std: 0.00689, params: {'n_neighbors': 189},
mean: 0.82870, std: 0.00649, params: {'n_neighbors': 190},
mean: 0.82878, std: 0.00704, params: {'n_neighbors': 191},
mean: 0.82665, std: 0.00624, params: {'n_neighbors': 192},
mean: 0.82718, std: 0.00654, params: {'n_neighbors': 193},
mean: 0.82478, std: 0.00895, params: {'n_neighbors': 194},
mean: 0.82550, std: 0.00846, params: {'n_neighbors': 195},
mean: 0.82230, std: 0.00974, params: {'n_neighbors': 196},
mean: 0.82274, std: 0.01041, params: {'n_neighbors': 197},
mean: 0.81936, std: 0.00926, params: {'n_neighbors': 198},
mean: 0.81972, std: 0.00963, params: {'n_neighbors': 199}]
In [104]:
# create a list of the mean scores only
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print(grid_mean_scores)
[0.9623966574806649, 0.9588407858476309, 0.9456840608054049, 0.94621744155036, 0.9387501111209885, 0.9428393634989777, 0.9379500400035559, 0.9383945239576851, 0.9335052004622633, 0.9349275491154769, 0.9304827095741843, 0.9325273357631789, 0.9293270512934483, 0.9296826384567517, 0.9271935283136279, 0.9281713930127122, 0.9255933860787625, 0.9272824251044537, 0.9249711085429816, 0.9247044181705041, 0.9220375144457285, 0.9218597208640769, 0.9190150235576495, 0.9177704684860877, 0.9156369455062672, 0.914570184016357, 0.9124366610365365, 0.9124366610365365, 0.911014312383323, 0.9111032091741488, 0.9090585829851542, 0.9094141701484576, 0.9077251311227664, 0.9072806471686372, 0.9055027113521202, 0.9041692594897324, 0.9029247044181705, 0.9009689750200017, 0.9003466974842208, 0.8987465552493555, 0.8975908969686194, 0.8962574451062316, 0.8953684771979732, 0.8927015734731976, 0.8917237087741132, 0.8895012890034669, 0.8890568050493377, 0.8866565916970397, 0.8857676237887813, 0.8852342430438261, 0.8845230687172193, 0.8837229975997867, 0.8828340296915281, 0.8814116810383145, 0.881144990665837, 0.8806116099208818, 0.879900435594275, 0.8791003644768424, 0.8789225708951907, 0.8780336029869322, 0.8776780158236288, 0.8768779447061961, 0.8765223575428926, 0.8759000800071117, 0.8755444928438083, 0.8747444217263757, 0.8741221441905948, 0.873944350608943, 0.8734109698639879, 0.8731442794915104, 0.8726997955373811, 0.8722553115832519, 0.871544137256645, 0.8707440661392124, 0.8705662725575607, 0.8700328918126056, 0.8696773046493022, 0.8675437816694818, 0.8669215041337007, 0.8656769490621389, 0.8659436394346164, 0.8646101875722286, 0.8638990132456218, 0.8623877678015823, 0.862565561383234, 0.8608765223575429, 0.8608765223575429, 0.8598097608676327, 0.8593652769135034, 0.8582985154235933, 0.8582096186327673, 0.8561649924437728, 0.8565205796070762, 0.8551871277446884, 0.8554538181171659, 0.853675882300649, 0.8533202951373455, 0.8519868432749578, 0.851897946484132, 0.8502089074584408, 0.8503867010400925, 0.8483420748510979, 0.848253178060272, 0.8455862743354965, 0.8455862743354965, 0.8441639256822828, 0.8445195128455862, 0.8437194417281536, 0.8439861321006311, 0.8425637834474176, 0.8423859898657658, 0.8413192283758556, 0.8408747444217264, 0.8403413636767713, 0.8401635700951195, 0.8398079829318161, 0.8397190861409903, 0.8386523246510801, 0.8389190150235577, 0.8388301182327318, 0.8389190150235577, 0.8366076984620855, 0.8370521824162148, 0.8366965952529114, 0.8371410792070406, 0.8373188727886923, 0.8374077695795181, 0.8373188727886923, 0.8374077695795181, 0.8375855631611698, 0.837496666370344, 0.8375855631611698, 0.8376744599519957, 0.8373188727886923, 0.8377633567428215, 0.8372299759978665, 0.8379411503244732, 0.8368743888345631, 0.8368743888345631, 0.8360743177171304, 0.8362521112987821, 0.8350075562272202, 0.8353631433905236, 0.8342963819006134, 0.8353631433905236, 0.835096453018046, 0.8352742465996977, 0.8341185883189617, 0.8349186594363943, 0.8347408658547426, 0.8348297626455685, 0.8349186594363943, 0.8353631433905236, 0.8349186594363943, 0.8349186594363943, 0.8349186594363943, 0.835096453018046, 0.8352742465996977, 0.8355409369721754, 0.8355409369721754, 0.8351853498088719, 0.8351853498088719, 0.8353631433905236, 0.8352742465996977, 0.8353631433905236, 0.8349186594363943, 0.835096453018046, 0.8352742465996977, 0.8349186594363943, 0.8345630722730909, 0.8344741754822651, 0.8337630011556583, 0.8337630011556583, 0.832696239665748, 0.8328740332473997, 0.8326073428749222, 0.8323406525024447, 0.8322517557116188, 0.8320739621299671, 0.8318072717574896, 0.8318961685483154, 0.8324295492932705, 0.8326073428749222, 0.8322517557116188, 0.8327851364565739, 0.8311849942217086, 0.831540581385012, 0.8294959551960174, 0.8300293359409725, 0.8286958840785847, 0.8287847808694107, 0.8266512578895902, 0.8271846386345453, 0.8247844252822473, 0.8254955996088541, 0.8222953151391235, 0.8227397990932528, 0.8193617210418704, 0.8197173082051737]
In [105]:
# plot the results (Manhattan distance)
plt.plot(k_potentials, grid_mean_scores)
plt.xlabel('Potential K values for KNN')
plt.ylabel('Cross-Validated Accuracy')
Out[105]:
<matplotlib.text.Text at 0x120cb5a50>

In [106]:
# plot the results (using Euclidean distance)
plt.plot(k_potentials, grid_mean_scores)
plt.xlabel('Potential K values for KNN')
plt.ylabel('Cross-Validated Accuracy')
Out[106]:
<matplotlib.text.Text at 0x12143b7d0>

In [107]:
# examine the best model
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
0.962396657481
{'n_neighbors': 1}
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
metric_params=None, n_jobs=1, n_neighbors=1, p=2,
weights='uniform')
In [ ]:

In [108]:
# read in the data & create matrices
X = X_scaled[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", \
"time_spend_company", "Work_accident", "promotion_last_5years", "low", "medium",\
"technical", "support", "IT", "product_mng", "marketing", "RandD", "accounting",\
"hr"]]
y = X_scaled["left"]
In [176]:
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
Support Vector Machines
In [177]:
# Train, fit and predict with SVM

from sklearn import svm
Model=svm.SVC(kernel='linear')
Model.fit(X_train,y_train)
Y_pred=Model.predict(X_test)
In [178]:
# Cross validation score for SVM
scores = cross_val_score(Model, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.780694712068
In [179]:
# Evaluate SVM for accuracy

from sklearn import metrics
metrics.accuracy_score(y_test,Y_pred)
Out[179]:
0.78106666666666669
In [180]:
# Confusion matrix for SVM

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2696  166]
[ 655  233]]
In [181]:
# Classification Accuracy confirmation between manual calculation and formula for SVM

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.781066666667
0.781066666667
In [182]:
# Misclassification Accuracy confirmation between manual calculation and formula for SVM
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.218933333333
0.218933333333
In [183]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated SVM

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.262387387387
0.262387387387
In [184]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for SVM

print(TN / float(TN + FP))
0.941998602376
In [185]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for SVM

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.583959899749
0.583959899749
In [186]:
# Misclassification Accuracy confirmation between manual calculation and formula for SVM
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.218933333333
0.218933333333
In [187]:
# Roc_curve for Support Vector Machine Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [92]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [93]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Support Vector Machines: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

Random Forest
In [157]:
# Random Forests Classifier from Scikit Learn

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_jobs=2)
RF.fit(X_train,y_train)
Y_pred=RF.predict(X_test)
In [158]:
# Score for 10-fold Cross Validation for Random Forests Classifier

scores = cross_val_score(RF, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.986309602501
In [159]:
# Prediction Accuracy for Random Forests

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.9896
In [160]:
# Confusion matrix for Random Forests

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2857    5]
[  34  854]]
In [161]:
# Classification Accuracy confirmation between manual calculation and formula for Random Forest

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.9896
0.9896
In [162]:
# Misclassification Accuracy confirmation between manual calculation and formula for Random Forest
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0104
0.0104
In [163]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for Random Forest Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.961711711712
0.961711711712
In [164]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for Random Forest Classifier

print(TN / float(TN + FP))
0.998252969951
In [165]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for Random Forest Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.994179278231
0.994179278231
In [166]:
# Misclassification Accuracy confirmation between manual calculation and formula for Random Forest
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0104
0.0104
In [167]:
# Roc_curve for Random Forest Model


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [168]:
y_pred_prob_Random_Forest = RF.predict_proba(X_test)[:, 1]
In [172]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_Random_Forest, bins=5)
plt.xlim(0, 1)
plt.title('Random Forest Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[172]:
<matplotlib.text.Text at 0x12013a8d0>

In [170]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
return
In [171]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Random Forest: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

Bagging
In [109]:
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
In [110]:
# Bagging Classifier

seed = 7
cart = DecisionTreeClassifier()
num_trees = 100
Bagging = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
Bagging.fit(X_train,y_train)
Y_pred=Bagging.predict(X_test)
In [111]:
# Cross Validation for Bagging

scores = cross_val_score(Bagging, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.987910076856
In [112]:
# Prediction Accuracy for Bagging

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.989333333333
In [113]:
# Confusion matrix for Bagging

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2854    8]
[  32  856]]
In [114]:
# Classification Accuracy confirmation between manual calculation and formula for Bagging

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.989333333333
0.989333333333
In [115]:
# Misclassification Accuracy confirmation between manual calculation and formula for Bagging
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0106666666667
0.0106666666667
In [116]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for Bagging Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.963963963964
0.963963963964
In [117]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for Bagging Classifier

print(TN / float(TN + FP))
0.997204751922
In [118]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for Bagging Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.990740740741
0.990740740741
In [119]:
# Misclassification Accuracy confirmation between manual calculation and formula for Bagging
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0106666666667
0.0106666666667
In [120]:
# Roc_curve for Bagging


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [121]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
In [122]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Bagging: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [123]:
y_pred_prob_Bagging = Bagging.predict_proba(X_test)[:, 1]
In [124]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_Bagging, bins=5)
plt.xlim(0, 1)
plt.title('Bagging Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[124]:
<matplotlib.text.Text at 0x11b217610>

In [ ]:

AdaBoost
In [125]:
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
In [126]:
# AdaBoost Classifier

seed = 7
cart = AdaBoostClassifier()
num_trees = 100
AdaBoostClassifier = AdaBoostClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
AdaBoostClassifier.fit(X_train,y_train)
Y_pred=AdaBoostClassifier.predict(X_test)
In [127]:
# Cross Validation for AdaBoost

scores = cross_val_score(AdaBoostClassifier, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.958576016468
In [128]:
# Prediction Accuracy for AdaBoost

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.958133333333
In [129]:
# Confusion matrix for AdaBoost

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2793   69]
[  88  800]]
In [130]:
# Classification Accuracy confirmation between manual calculation and formula for Bagging

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.958133333333
0.958133333333
In [131]:
# Misclassification Accuracy confirmation between manual calculation and formula for AdaBoost
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0418666666667
0.0418666666667
In [132]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for AdaBoost Boosting Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.900900900901
0.900900900901
In [133]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for AdaBoost Classifier

print(TN / float(TN + FP))
0.975890985325
In [134]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for AdaBoost Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.920598388953
0.920598388953
In [135]:
# Misclassification Accuracy confirmation between manual calculation and formula for AdaBoost
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0418666666667
0.0418666666667
In [136]:
# Roc_curve for AdaBoost


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [137]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
In [138]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='AdaBoost: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [139]:
y_pred_prob_AdaBoost = AdaBoostClassifier.predict_proba(X_test)[:, 1]
In [140]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_AdaBoost, bins=5)
plt.xlim(0, 1)
plt.title('AdaBoost Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[140]:
<matplotlib.text.Text at 0x11b367a10>

In [ ]:

In [ ]:

In [ ]:

Gradient Boosting
In [141]:
from sklearn.ensemble import GradientBoostingClassifier
In [142]:
# Train and predict Gradient Boosting

GradientBoosting = GradientBoostingClassifier()
GradientBoosting.fit(X_train,y_train)
Y_pred=GradientBoosting.predict(X_test)
In [143]:
# Cross Validation for Gradient Boosting

scores = cross_val_score(GradientBoosting, X_train, y_train, cv=10, scoring= 'accuracy')
print(scores.mean())
0.976443246546
In [144]:
# Prediction Accuracy for Gradient Boosting

from sklearn import metrics
print(metrics.accuracy_score(y_test, Y_pred))
0.974133333333
In [145]:
# Confusion matrix for Gradient Boosting

confusion = metrics.confusion_matrix(y_test, Y_pred)
print confusion
[[2832   30]
[  67  821]]
In [146]:
# Classification Accuracy confirmation between manual calculation and formula for Gradient Boosting

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, Y_pred))
0.974133333333
0.974133333333
In [147]:
# Misclassification Accuracy confirmation between manual calculation and formula for Gradient Boosting
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0258666666667
0.0258666666667
In [148]:
# This Represents the Sensitivity, the % out of True positives that were correctly evaluated as such.
# This is calculated for Gradient Boosting Classifier

print(TP / float(TP + FN))
print(metrics.recall_score(y_test, Y_pred))
0.92454954955
0.92454954955
In [149]:
# This represents the Specificity, the % of total True negatives that were correctly identified as negatives.
# This is calculated for Gradient Classifier

print(TN / float(TN + FP))
0.989517819706
In [150]:
# This represents the precision, what % of the values that were predicted to be positive actually are positive.
# This is calculated for Gradient Boosting Classifier.

print(TP / float(TP + FP))
print(metrics.precision_score(y_test, Y_pred))
0.964747356052
0.964747356052
In [151]:
# Misclassification Accuracy confirmation between manual calculation and formula for Gradient Boosting
# This is also simply equal to 1-(Classification Accuracy calculated above).

print((FP + FN) / float(TP + TN + FP + FN))
print(1 - metrics.accuracy_score(y_test, Y_pred))
0.0258666666667
0.0258666666667
In [152]:
# Roc_curve for Gradient Boosting


from sklearn.metrics import roc_curve, auc
plt.style.use('fivethirtyeight')

FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(y_test, Y_pred)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (employee left)
plt.figure(figsize=[5,5])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0, 1])
plt.ylim([0, 1.5])
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Gradient Boosting Receiver operating characteristic for employees leaving', fontsize=18)
plt.legend(loc="upper left")
plt.show()

In [153]:
import matplotlib.patheffects as path_effects
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, \
labels = ['Left', 'Stayed']):

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=cmap)

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, fontsize = 12)
plt.yticks(tick_marks, labels, rotation = 90, fontsize = 12)

plt.title(title, fontsize = 24)
plt.ylabel('True', fontsize = 18)
plt.xlabel('Predicted', fontsize = 18)
plt.tight_layout()

width, height = cm.shape

for x in xrange(width):
for y in xrange(height):
plt.annotate(str(cm[x][y]), xy=(y, x), 
horizontalalignment='center',
verticalalignment='center',
color = 'white',
fontsize=18).set_path_effects([path_effects.Stroke(linewidth=1, \
foreground='black'), path_effects.Normal()])
In [154]:
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, Y_pred)


plot_confusion_matrix(cm, title='Gradient Boosting: Confusion Matrix', \
labels=['Stayed', 'Left'], cmap=plt.cm.Greens)

In [155]:
y_pred_prob_GradientBoosting = GradientBoosting.predict_proba(X_test)[:, 1]
In [156]:
# histogram of predicted probabilities of people leaving the company 
plt.hist(y_pred_prob_GradientBoosting, bins=5)
plt.xlim(0, 1)
plt.title('Gradient Boosting Histogram of predicted probabilities')
plt.xlabel('Predicted probability of person leaving')
plt.ylabel('Frequency')
Out[156]:
<matplotlib.text.Text at 0x11c2566d0>

In [ ]:

In [ ]:

Part IV
Conclusion
One interesting chart showed us which departments have the most people leaving (and staying). They are Sales followed by Technical and Support. That means that the biggest effort at the company should be placed on keeping the strong employees in those departments (but by no means neglect the other departments).
There are several other interesting patterns in the data. Some are very intuitive whereas others very surprising and unexpected. There are three clusters of people leaving the company (the most important cluster is not as dense, in the sense that there are actually slightly more people there that stay than leave, but nevertheless its the most interesting and surprising cluster).
The three groups are employees with high performance and low satisfaction, employees with low satisfcation and performance (rather obvious) and employees with high satisfaction and performance (a surprising result). The first two results are very strong but rather trivial, there is limited use in providing detailed analyses of them. However, the somewhat less dense but very significant group of highly satisfied employees that do remarkably well in their evaluations is of extreme importance. Why do they leave? Well, there are several important factors to consider.
The people with high salaries and at even average satisfaction levels rarely leave the firm. Hence, one attribute that people who tend to leave will generally possess is salaries that are not high and fall in the low or medium range. It is interesting to note though that people with near perfect to perfect satisfaction levels do not leave even if they are in the low or medium income range brackets. Hence, a possible course of action to potentially alleviate this issue is increase salaries for good employees. Also, a further investigation into what might be the differentiating factor between people who are very satisfied (0.7-0.9 rate) vs super satisfied (0.9-1.0) can be done to see if it is possible to further increase the satisfaction level of those that are very satisfied but not perfectly so (maybe its possible to do this while at the same time saving money by not instituting as much pay increases if there is a way to get them satisfied in another manner, perhaps by working less hours for instance) would be a good idea.
Secondly, promotions are a huge factor. Looking at the plot of promotions during the past five years, it is very clear that people who are promoted stay and people who are not promoted leave. Of course, promotions and salary increases have some correlation because people tend to receive salary increases during promotions. Hence, it's important to promote employees if they do good work! That will help decrease the turnover rate.
The third important point is that employees who work a lot of hours that stay at the company for at least a medium amount of time tend to leave. Only the relatively small number of employees who work at the company for many years stay at the company regardless of working hours but very few make it there. Hence, the hours worked plays an important decision making role when employees think about whether to stay at the firm or seek other employment. Due to this finding it is important to see if any kind of rebalancing can be done so that part of the work can perhaps be shifted between employees who do not work as many hours and those that do in order to make the distribution of hours worked by employees more uniform which can perhaps increase employees willingness to stay at the company longer. It is quiet possible that some employees just feel burnt out even if they are generally happy with their job. Due to an inefficient work life balance, they might seek to find employment elsewhere.
Multiple machine learning algorithms were ran on the data and quiet a few did relatively well in terms of making predictions on out of sample data as measured by the accuracy score, the confusion matrix and deriatives thereof as well as the ROC Curve. In particular, the Random Forest and Bagging algorithms did the best with accuracy and cross-validation scores marginally different from each other just under 99%. Both correctly predicted over 3,700 out of 3,750 employees accurately (compared to a baseline accuracy of only 76% (2,850) as to whether they would leave/stay). The various scores computed from the confusion matrix were all almost perfect. Particularly importantly, the ROC curve has an area covering of .98 for both those algorithms.
Hence, using either of these two algorithms (Random Forests or Bagging) have given us great predictions as to which employees will leave vs which ones would stay.
Assumptions
We have assumed the dataset is accurate as we have not audited its validity.
Another assumption we made is that we trained on 75% of the data, conducted cross-validation, and then tested on the 25% remaining out of sample. However, when trying to sample to train a larger sample of the data (such as 90/10 test/train split), we received very similar results in our algorithms. Hence, our model is robust as should perform well when used on future data.
An assumption made by the Random Forest Model which did exceptionally well is that the sampling that it conducted during boostrap aggregation is actually representative. This is a safe assumption to make and the model has done really well on out of sample data.



![Graph]({{ site.url }}/images/Graph_for_project.png)
