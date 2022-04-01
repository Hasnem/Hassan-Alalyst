#!/usr/bin/env python
# coding: utf-8

# In[278]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Objective: 
# **Make a model to predict the app rating, with other information about the app provided.**

# # Problem Statement:
# 
# **Google Play Store team is about to launch a new feature wherein, certain apps that are promising, are boosted in visibility. The boost will manifest in multiple ways including higher priority in recommendations sections (“Similar apps”, “You might also like”, “New and updated games”). These will also get a boost in search results visibility.  This feature will help bring more attention to newer apps that have the potential.**

# In[279]:


#Load the data file using pandas. 
google_play = pd.read_csv('googleplaystore.csv')


# In[280]:


google_play.head()


# In[281]:


google_play.info()


# # Analysis to be done: 
# **The problem is to identify the apps that are going to be good for Google to promote. App ratings, which are provided by the customers, is always a great indicator of the goodness of the app. The problem reduces to: predict which apps will have high ratings.**

# In[282]:


#Check for null values in the data. Get the number of null values for each column.
google_play.isnull().sum()


# In[283]:


google_play = google_play.dropna()


# In[284]:


#Size column has sizes in Kb as well as Mb. To analyze, we will need to convert these to numeric.
google_play['Size'] = google_play['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)
google_play['Size'] = google_play['Size'].apply(lambda x: str(x).replace('k', '') if 'k' in str(x) else x)
google_play['Size'] = google_play['Size'].apply(lambda x: float(str(x).replace('M', '')) * 1024 if 'M' in str(x) else x)
google_play['Size'] = google_play['Size'].apply(lambda x: float(x))


# In[285]:


#Drop NaN from Size.
google_play = google_play.dropna()


# In[286]:


#Convert Reviews to numeric
google_play['Reviews'] = google_play['Reviews'].apply(lambda x: int(x))
print(google_play['Reviews'].dtypes)


# In[287]:


#remove ‘+’, ‘,’ from the field, convert it to integer
google_play['Installs'] = google_play['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)
google_play['Installs'] = google_play['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)
google_play['Installs'] = google_play['Installs'].apply(lambda x: int(x))
print(google_play['Installs'].dtypes)


# In[288]:


#Remove ‘$’ sign, and convert Price to numeric
google_play['Price'] = google_play['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
google_play['Price'] = google_play['Price'].apply(lambda x: float(x))
print(google_play['Price'].dtypes)


# In[289]:


#Drop rating that have a value outside 1-5 range.
google_play = google_play[(google_play.Rating >0) | (google_play.Rating<=5)]


# In[290]:


#Reviews should not be more than installs as only those who installed can review the app.
google_play = google_play[google_play.Installs>=google_play.Reviews]


# In[326]:


#For free apps (type = “Free”), the price should not be >0.
google_play[(google_play.Type == 'Free') & (google_play.Price>0)]
# there are no records


# In[292]:


paid_apps = google_play[google_play.Price>0]


# In[293]:


# Boxplot for Price
plt.boxplot(paid_apps.loc[:,'Price'])
#plt.ylim([0,200])
plt.show()
#outliers are somewhere after 200


# In[294]:


#Boxplot for Reviews
sns.boxplot(google_play.Reviews)
#there aren't any apps with very high number of reviews,the values don't seem to be right?
#will do more analysis to see how reviews are distributed 


# In[295]:


#Histogram for Rating
plt.figure(figsize=(8,4))
sns.histplot(data=google_play['Rating'],bins=5)
#ratings are distributed more toward higher ratings.


# In[296]:


##Histogram for Size
sns.histplot(google_play.Size)


# # Outlier treatment
# 

# In[297]:


#Price: From the box plot, it seems like there are some apps with very high price. 
#A price of $200 for an application on the Play Store is very high and suspicious!
#Drop apps with price more than 200 as most seem to be junk apps
google_play=google_play[google_play['Price']<=200]


# In[298]:


#Reviews: Very few apps have very high number of reviews. 
#These are all star apps that don’t help with the analysis and, in fact, will skew it. 
#Drop records having more than 2 million reviews.
google_play=google_play[google_play['Reviews']<=2000000]


# In[299]:


#Installs:There seems to be some outliers in this field too. 
#Apps having very high number of installs should be dropped from the analysis.
#Find out the different percentiles – 10, 25, 50, 70, 90, 95, 99
print(np.percentile(google_play.Installs,[10,25,50,70,90,95,99]))


# In[300]:


#Decide 95 percentaile to be as cutoff for outlier and drop records having values more than that.
google_play=google_play[google_play.Installs<= np.percentile(google_play.Installs,95)]


# # Bivariate analysis
# **Let’s look at how the available predictors relate to the variable of interest, i.e., our target variable rating.we are going to make scatter plots (for numeric features) and box plots (for character features) to assess the relations between rating and the other features**

# In[301]:


#joinplot for Rating vs. Price
plt.figure(figsize=(10,5),dpi=200)
sns.jointplot(x=google_play[(google_play['Type']=="Paid") & (google_play['Price']<50)]['Price'], 
              y=google_play[(google_play['Type']=="Paid") & (google_play['Price']<50)]['Rating'])


# In[302]:


google_play['Rating'].corr(google_play['Price'])
#we notice that price and rating are not correlated


# In[303]:


#scatter plot for Rating vs. Size
sns.jointplot(x=google_play['Size'],y=google_play['Rating'],kind="hex", color="#6EA3CC")
#we notice that price and Size are not realy correlated
#heavier apps are not rated better


# In[304]:


google_play['Rating'].corr(google_play['Size'])


# In[305]:


#scatter plot for Rating vs. Reviews
plt.scatter(y=google_play['Rating'],x=google_play['Reviews'])
plt.xlabel('Reviews')
plt.ylabel('Rating')


# In[306]:


google_play['Rating'].corr(google_play['Reviews'])
#Reviews and Ratings are weakly correlated 
# it seems like apps with high number of reviews are most likly to get a better rating.


# In[307]:


#boxplot for Rating vs. Content Rating
plt.figure(figsize=(12,6))
sns.boxplot(y='Rating',x='Content Rating',data=google_play)
#Adults only 18+ has slightly butter rating average than other Content types


# In[308]:


google_play.groupby('Content Rating').mean()['Rating']


# In[309]:


#boxplot for Ratings vs. Category
plt.figure(figsize=(12,6),dpi=200)
sns.boxplot(y='Rating',x='Category',data=google_play)
plt.xticks(rotation=70,fontsize=7)
plt.show()
#EVENTS genra has the best rating.


# In[310]:


google_play.groupby('Category').mean()['Rating'].nlargest()


# # Data preprocessing
# - For the steps below,we will create a copy of the dataframe to make all the edits.
# 
# - 'Reviews' and 'Install' have some values that are still relatively very high. Before building a linear regression model, we will   need to reduce the skew. 
# 
# - Drop columns 'App', 'Last Updated', 'Current Ver', and 'Android Ver'. These variables are not useful for our task.
# 
# - Get dummy columns for 'Category', 'Genres', and 'Content Rating'. This needs to be done as the models do not understand categorical data, and all data should be numeric.

# In[311]:


inp1=google_play.copy() #create a copy


# In[312]:


#Apply log transformation (np.log1p) to Reviews and Installs to reduce skewness.
inp1['Reviews']= np.log1p(inp1.Reviews)
inp1['Installs']= np.log1p(inp1.Installs)


# In[313]:


#Drop unusefull columns fro out task
inp1= inp1.drop(['App', 'Last Updated','Current Ver','Android Ver'], axis=1)


# In[314]:


#convert character fields to numeric using dummy incoding
inp2 = pd.get_dummies(inp1,prefix_sep='_',columns=['Type','Category', 'Genres', 'Content Rating'])
inp2.iloc[0:100,:]


# # Multiple Linear regssion Model building

# In[315]:


y = inp2['Rating']
x= inp2.iloc[:,1:]
# Train and test split(70-30)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =  train_test_split(x,y,test_size=.30)


# In[316]:


from sklearn.linear_model import LinearRegression  #import linear regression


# In[317]:


model_inp2 = LinearRegression().fit(x_train,y_train) #bulding model
predictions_Rating = model_inp2.predict(x_test) ## Making predictions


# In[318]:


google_test_combined = pd.concat([x_test.reset_index(drop=True),
                                   y_test.reset_index(drop=True),
                                   pd.DataFrame(predictions_Rating,columns=['Predicted_Rating'])],axis=1)
#combin and reset indexes f x_test,y_test,and predicted ratings in one datafram
google_test_combined.Predicted_Rating = round(google_test_combined.Predicted_Rating,1)
google_test_combined


# In[319]:


#calculate error pct
google_test_combined['Err_pct'] = abs(google_test_combined.Rating - 
                                      google_test_combined.Predicted_Rating)/google_test_combined.Rating


# In[320]:


# Error Rate
google_test_combined['Err_pct'].mean()


# In[321]:


# Accuracy Rate
1- google_test_combined['Err_pct'].mean()


# In[322]:


from sklearn.metrics import r2_score #import R2


# In[323]:


r2_score(google_test_combined.Rating,google_test_combined.Predicted_Rating) # r2_score

