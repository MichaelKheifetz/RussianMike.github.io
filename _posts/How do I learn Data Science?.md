---
layout: page
title: How do you learn Data Science?

---

This post is mainly intended for those early in their Data Science journey. Primarily those that are very excited about the field but have not really done anything or very little in it. There are so many things to learn in Data Science that it would be both quite challenging to cover everything in one post, and not necessary to do so because there are many such questions asked and answered in various sources online. Towards the end of this post, I will point out some of those. 


# Introduction

So to be an advanced Data Scientist, there are a ton of things you have to know, a lot of very strong grounding in the undergrad Mathematics (particularly Linear Algebra, Probability Theory and Calculus) as well as Statistics is necessary. You should have a solid understanding of data structures/algorithms and be able to code them up, preferably in Python and/or R, knowledge in big data technologies (Hadoop, Spark, etc). This list can go on for a while.

The main aim of my post is to illustrate the fundamentals of the Data Science workflow to someone who is a complete beginner in data analysis into the field (especially without a rock solid mathematical background. By no means do I imply that a solid mathematical/statistical background is not necessary, you can just accomplish a lot of things early on without having deep mathematical training) so they can gain a strong familiarity with the entire Data Science workflow, and in a rather short amount of time, be able to conduct a data analysis project on their own, and believe me, that's some really cool stuff!

I think it is best for someone excited about data analysis and starting out to create a project almost right from the start (before digging deep into the really cool theory or advanced coding skills just so they can get a good feel for what the field is about and actually have a final working product that they themselves created!). Ideally of course, the more Python or R skills you have, the better, but you can start out with relatively little coding skills and still build a cool model with great accuracy that generalizes well to out of sample data, particularly with relatively simple datasets that don't have too many features that need to be dealt with in unique ways (more advanced stuff).

So my recommended course of action is to first take a quick Python course that can be completed in a relatively short period of time online (such as the very first of the sequence offered by Coursera here https://www.coursera.org/specializations/python) or any other introductory free online course in Python or R (there are countless other resources such as Udemy, Code Academy, EdX, etc). Also, make sure you download Jupyter Notebook (previously called IPython Notebook) so that you will be able to do your work there. You can easily run any code there and immediately run it, graph fancy visualizations, run models, and do pretty much anything. It looks very cool, nice and professional and you can later post your project up on Github for anyone to see.

After you take the Python class, you can take an introductory class in Statistics and read up about the key models in Machine Learning (such as Linear and Logistic Regressions, Random Forests, Bagging, Boosting, etc) to get an intuitive understanding of how they work. One of my personal favorites is Introduction to Statistical Learning by Hastie, Tibshirani (top researchers in this field and Professors at Stanford), they do a very good job introducing the material in their lectures and free textbook available online. (https://lagunita.stanford.edu/courses/HumanitiesSciences/StatLearning/Winter2016/about)


# The Fundamental steps in the Data Science
# workflow process


Now is the fun time, let's get to work. There are a few resources I found very valuable to help learning the entire Data Science workflow and help me better understand the workflow in the typical (not that there is much of such a thing in Data Science!) data analysis project. I recommend you to take a look at this post by Randal Olson, (an awesome Data Scientist at University of Pennsylvania with some really cool and insightful blog posts you can review later): 

https://github.com/rhiever/Data-Analysis-and-Machine-Learning-Projects/blob/master/example-data-science-notebook/Example%20Machine%20Learning%20Notebook.ipynb

In this post he goes over the entire data science workflow on the very basic Iris dataset that is very commonly used in introductory data science materials. So let's know go over some of these vary important steps you should engage in your first data analysis project.

What I personally recommend is that you review his or other blog posts in detail and try to use his work to help you with your thought process and start working on another dataset. Of course, there will be potentially different issues and the way you will have to deal with them will need to be quite different (because the relationships between features in the dataset can be completely different, you might need to impute the missing values using some methodology instead of dropping the null values, conduct Principal Component Analysis (PCA) to reduce your high dimensional dataset to few dimensions that capture most of the variance, etc, etc...) but the thought process in analyzing the data and the workflow provided in his blogpost provides good guidance on how to conduct a solid data project as well as code you can use in Python that can get you there (if you are using Python, the relevant libraries for most analyses that are necessary are Numpy, Pandas, Matplotlib and Scikit-Learn).

# What question do you want answered?

Before actually doing anything with the data itself, you have to be clear to yourself about what question you want to answer. Do you just want to explore the relationships between some of the features (variables) or are you trying to predict or classify one of the features into several different categories (the output variable)? How are you going to measure the success of your project once completed? For example, are you going to use a confusion matrix and ROC curve to evaluate your false positive and false negative rates or R^2 to measure your prediction accuracy for regression models?

# Check for outliers

So the first step in the Data Science process is cleaning your dataset so it is then easy to conduct exploratory data analysis and build a predictive model. So you want to identify potential problem areas with your dataset as your first step. Check to see if your target variable has some very obvious outliers by running a histogram and checking what the distribution is like. If it does, depending on why they are there, you might want to either delete them or keep them in the dataset (this is actually a rather challenging topic that would need a whole post onto itself, but there is no exact answer here, it will really depend on the dataset).

# Check for missing values

After you do this, you should check for missing values in the dataset. Here, it becomes both an art and a science. There is not necessarily only one correct way of addressing null values but there are are certaintly wrong ways to go about it. If your dataset contains 50,000 rows and you have 50 missing values, unless you are dealing with an extremely imbalanced dataset (for example, something involving cancer research where the classes are divided between over 99% and under 1% between cancerous and non-cancerous), you can generally drop these values without being concerned as they will have a negligible impact on your model's predictive power. However, if 20% of your dataset contains missing values, deleting them will make you lose a significant portion of your dataset in a non-random way and regardless of the model you will end up using, your model will lose significant predictive power because you will not be able to train and fit on the portion of the data that you delete. So what you need to do in such a scenario is impute your missing data using either the mean (the easiest and worst method but the fastest) or better yet, another methodology, for instance using another ML algorithm such as KNN (K Nearest Neighbors) or one of many other advanced techniques. 

# Make sure all your datatypes are of correct type to normalize and to build your model

A critical step in the data cleaning process is to than convert all the columns data types if they are not currently in integer or float format. In order to later preprocess and normalize your dataset, all your data (all the values in all the columns) must be numerical data types (can not be strings), hence you will need to code them as such. This will take all sorts of data cleaning such as removing potential strings from your dataset in order for it to only contain only numerical values. 

# Normalizing your dataset

When you are done with this step, you can normalize your data using one of several methodologies. One common way is by using the Standard Scaler in Python (this is the Z-score statistic that transfers all your dataset on a -3 to +3 scale). This is done because you want all your features to be reflected on the same scale. You do not want to compare small numerical values in a feature such as pounds a car weighs (thousands) vs the length of the car (a few meters) because that will create a completely skewed model, particularly if it uses a parametric method like linear regression. Here is the code that would run the MinMax Scaler in Python (MinMax Scaler scales everything on a 0 to 1 range):

from sklearn.preprocessing import MinMaxScaler
X_scaled = MinMaxScaler().fit_transform(HR.astype(float))
X_scaled = pd.DataFrame(X_scaled, columns = HR.columns)


# Exploratory Data Analysis

So the purpose of this very important step is to check the relationship between your features and your target variable that you will end up using in building a classification/regression model. This is best done by visualization of the relationship between different features. The kind of visualizations you use is a combination of your personal preferences, what particular dataset you are working with and what kind of audience you will be presenting your results to (some graphs are better for non-technical audiences. For example, histograms and bar charts are easy to follow for the non-technical audience vs boxplots which should be only presented to technical audiences, etc). Some really common visualizations are histograms, bar plots, scatter plots, correlation heatmaps and many others. The main purpose of this step is to see if you can detect any really significant correlations and potential insights in your data that you can than present and potentially provide recommendations on. This step is also very helpful for Predictive Modeling because it will help you give a visual grasp of which features are very important and which are not (and it what ways, e.g, positively vs negative correlated). You can than think about how to feature engineer your model more effectively to produce better accuracy scores. Here is one of the scatterplots I did in my project to get a better visual grasp of how different variables relate to one another:

![Satisfaction(/images/satisfaction_vs_evaluation.png)

# Predictive Modeling (Machine Learning)

This is arguably the most exciting part of Data Science (well in my opinion anyway). This is where you will build your classification/regression model that predicts/classifies your target variable. The first step is to decide on a train/test split (Usually either 70/30 or 80/20 are very commonly used ones) train the data, and predict using one of the Machine Learning models. You can try using many models, but aside from getting accuracy scores, you need to check cross validation to see that your model does not overfit. Other tests to see how good your model actually is include the various Confusion Matrix metrics such as Precision, Specificity and Sensitivity that get reflected in the ROC curve. The closer your model is to a 1 on a 0 to 1 scale, the better your model is (well if you are dealing with a classification problem that is). If you working on a regression problem, you want to check your Adjusted R^2 and your Root-Mean-Square-Error, Weighted Absolute Error, etc. These will give you a good idea of how good your model is. If you are not getting good accuracy or other metric scores, it is quite likely that you need to further enhance your model by feature selection & feature engineering. These are very deep subjects that have books written upon them so I will only mention them briefly in this post. It is quite likely that some of your input variables are much more correlated to your output variables then others. You can potentially improve your model by removing some of the features that explain very little of the outcome. This can be done using techniques such as PCA (where you keep the variables explaining most of the variance in the model), or something like forward or backward feature selection where you start out with no variables in your model and add a variable individually to the model only when it satisfied some criterion metric (such as a certain p-value) or vice-versa with backward selection where you start with the full model and drop variables that have little impact on your output. Feature engineering is largely an art in addition to being a science. It involves the creation of new unique variables that are combination of other variables in creative ways that when combined can show significant predictive power. For example, if your dataset contains people's weights and heights, you can add a feature called BMI (weight/height) which can give you better predictive power then before you added this feature to your model.

# Conclusion

Based on all this wonderful work that you did in the steps above, you should describe your conclusions and recommendations. Recommendations are very important as they are one of the main reasons that you have undertaken your project; to discover interesting and important insights in the data. It is also very important to explain the assumptions that your models and other parts of your project take, and to address whether these assumptions were valid to make in your particular situation. Another important element that you can include here is potential further research you can do related to this subject if you have more time such as develop more complicated models to improve your accuracy or get additional data that will make your models work better.

Finally you are done working on your first project! Horray!!

Here are some additional resources I recommend reviewing on what you need to know for Data Science to get closer to the mastery level:

https://www.quora.com/How-do-I-learn-data-science-in-a-time-efficient-manner

https://www.quora.com/What-are-the-best-resources-for-learning-how-to-use-Python-for-Machine-Learning-Data-Science

More Advanced Material:

https://www.quora.com/What-are-the-mathematical-pre-requisites-for-studying-machine-learning










