#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from dmba import classificationSummary
from sklearn import tree

xlsx = pd.ExcelFile('ebayAuctions.xlsx')
ebay_df = pd.read_excel(xlsx, 'eBay auctions')

# Data Cleaning / Exploratory Data Analysis
ebay_df.head()

# pretty well-balanced data set
ebay_df['Competitive?'].value_counts()
sns.countplot(ebay_df['Competitive?'])

# most auctions are related to music/movie/game or collectibles or toys/hobbies
ebay_df['Category'].value_counts()

# majority are US
ebay_df['Currency'].value_counts()

# there are no null values
ebay_df.info()

# confirming that there are no null values
ebay_df.isna().sum(axis = 'rows')

# quite a few duplicates - they could be multiple of the same item being put on ebay
sum(ebay_df.duplicated())

df_duplicate = ebay_df.duplicated()

# most of duplicates, if they are duplicates, are in music/movie/game, toys/hobbies categories
ebay_df[df_duplicate]['Category'].value_counts()

# hypothetically if we were to drop the duplicates, although they could be meaningful since a person could post multiple
# auctions for the same item on the same day
ebay_df_no_duplicates = ebay_df.drop_duplicates(keep = 'first')

ebay_df_no_duplicates.head()

ebay_df_no_duplicates.info()

# still a decently balanced data set but not as balanced
ebay_df_no_duplicates['Competitive?'].value_counts()

# comparing competitive auctions by category - with duplicates
# some categories may be more competitive than others
counts = (ebay_df.groupby(['Competitive?'])['Category'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
plt.figure(figsize=(15,10))
sns.barplot(x = 'Category', y ='percentage', hue = 'Competitive?', data = counts)
plt.xlabel('Category?')
plt.legend(title = 'Competitive?')
plt.xticks(rotation = 90)

# comparing competitive auctions by category - with no duplicates
counts = (ebay_df_no_duplicates.groupby(['Competitive?'])['Category'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
plt.figure(figsize=(15,10))
sns.barplot(x = 'Category', y ='percentage', hue = 'Competitive?', data = counts)
plt.xlabel('Category?')
plt.legend(title = 'Competitive?')
plt.xticks(rotation = 90)

# comparing competitive auctions by currency price - with duplicates
counts = (ebay_df.groupby(['Competitive?'])['Currency'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
plt.figure(figsize=(15,10))
sns.barplot(x = 'Currency', y ='percentage', hue = 'Competitive?', data = counts)
plt.xlabel('Currency?')
plt.legend(title = 'Competitive?')

# comparing competitive auctions by currency price - without duplicates
# currency doesn't seem to make a difference
counts = (ebay_df_no_duplicates.groupby(['Competitive?'])['Currency'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
plt.figure(figsize=(15,10))
sns.barplot(x = 'Currency', y ='percentage', hue = 'Competitive?', data = counts)
plt.xlabel('Currency?')
plt.legend(title = 'Competitive?')

# comparing competitive auctions by endDay - with duplicates
# monday seems to be a good day to end your auction
counts = (ebay_df.groupby(['Competitive?'])['endDay'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
plt.figure(figsize=(15,10))
sns.barplot(x = 'endDay', y ='percentage', hue = 'Competitive?', data = counts)
plt.xlabel('endDay?')
plt.legend(title = 'Competitive?')

ebay_df['Duration'] = ebay_df['Duration'].astype("object")

# comparing competitive auctions by duration - with duplicates
# auctions open for 5 days seem to be more competitive, 7 as well but has a higher percentage of not competitive
counts = (ebay_df.groupby(['Competitive?'])['Duration'].value_counts(normalize = True).rename('percentage')
         .mul(100).reset_index())
plt.figure(figsize=(15,10))
sns.barplot(x = 'Duration', y ='percentage', hue = 'Competitive?', data = counts)
plt.xlabel('Duration')
plt.legend(title = 'Competitive?')

# OpenPrice
fig, ax = plt.subplots()
sns.boxplot(x=ebay_df['Competitive?'], y=ebay_df['OpenPrice'])
plt.xticks(rotation = 90)
plt.xlabel("Competitive?")
ax.set_ylim(-5,50)

# there are open prices that are higher than 100, cut the x-axis down just to make it more interpretable
ebay_df_comp = ebay_df[ebay_df['Competitive?'] == 1]
fig, ax = plt.subplots()
sns.histplot(data = ebay_df_comp, x = 'OpenPrice')
ax.set_xlim(-5,100)

# there are open prices that are higher than 100, cut the x-axis down just to make it more interpretable
# looks like auctions that are not competitive have higher opening prices
ebay_df_comp = ebay_df[ebay_df['Competitive?'] == 0]
fig, ax = plt.subplots()
sns.histplot(data = ebay_df_comp, x = 'OpenPrice')
ax.set_xlim(-5,100)

# Rating
fig, ax = plt.subplots()
sns.boxplot(x=ebay_df['Competitive?'], y=ebay_df['sellerRating'])
plt.xticks(rotation = 90)
plt.xlabel("Competitive?")
ax.set_ylim(-5,25000)

# sellerRating histogram for competitive auctions
ebay_df_comp = ebay_df[ebay_df['Competitive?'] == 1]
fig, ax = plt.subplots()
sns.histplot(data = ebay_df_comp, x = 'sellerRating')

# sellerRating histogram for not-competitive auctions
ebay_df_comp = ebay_df[ebay_df['Competitive?'] == 0]
fig, ax = plt.subplots()
sns.histplot(data = ebay_df_comp, x = 'sellerRating')

# Code for the boxplots panel:
fig, axes = plt.subplots(3, 6, sharey=True, figsize=(16,12))
for c,i in zip(ebay_df.Category.unique(),np.arange(18)):
    sns.boxplot(ax=axes[i%3,i//3], x=ebay_df[ebay_df['Category']==c]['Competitive?'], y=ebay_df[ebay_df['Category']==c]['sellerRating'])
    plt.xticks(rotation = 90)
    axes[i%3,i//3].set_xlabel(c)
    plt.yscale('log')

# Code for the boxplots panel:
fig, axes = plt.subplots(3, 6, sharey=True, figsize=(16,12))
for c,i in zip(ebay_df.Category.unique(),np.arange(18)):
    sns.boxplot(ax=axes[i%3,i//3], x=ebay_df[ebay_df['Category']==c]['Competitive?'], y=ebay_df[ebay_df['Category']==c]['OpenPrice'])
    plt.xticks(rotation = 90)
    axes[i%3,i//3].set_xlabel(c)
    plt.yscale('log')

# Data Engineering

# Keep duplicates in the data set - technically sellers on ebay can put multiple auctions for the same item so cannot
# tell for sure whether they are duplicates


pd.get_dummies(ebay_df['Category'], prefix = 'category')
pd.get_dummies(ebay_df['Currency'], prefix = 'currency')
pd.get_dummies(ebay_df['Duration'], prefix = 'duration')
pd.get_dummies(ebay_df['endDay'], prefix = 'endDay')

# creating final table with dummy variables
ebay_df_final = pd.concat([ebay_df, pd.get_dummies(ebay_df['Category'], prefix = 'category'),  pd.get_dummies(ebay_df['Currency'], prefix = 'currency'), pd.get_dummies(ebay_df['Duration'], prefix = 'duration'),pd.get_dummies(ebay_df['endDay'], prefix = 'endDate')], axis = 1)
ebay_df_final.drop(['Category', 'Currency', 'Duration', 'endDay'], axis = 1, inplace = True)
ebay_df_final.info()

# declare X dataframe
X = ebay_df_final.drop(['Competitive?'], axis = 1)
y = ebay_df_final['Competitive?']

# splitting the data for the first decision tree with all predictors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)


fullClassTree = DecisionTreeClassifier(min_samples_leaf = 50, random_state = 42)
fullClassTree.fit(X_train, y_train)
y_predicted = fullClassTree.predict(X_test)
fullClassTree.score(X_test, y_test)

fullClassTree.get_params()
classificationSummary(y_train, fullClassTree.predict(X_train))
fullClassTree.tree_.node_count
fullClassTree.tree_.max_depth

plt.figure(figsize=(25,20))
tree.plot_tree(fullClassTree, feature_names = X.columns, class_names = ['NC', 'C'])
export_graphviz(fullClassTree, out_file = 'fullClassTree_v1.dot', feature_names = X_train.columns)
# Can visualize the .dot file by copying the text into: http://webgraphviz.com/

# performing a grid search
param_grid = {
    'max_depth' : [5,10,15],
    'min_samples_split':[10, 20, 30, 40, 50, 60, 70, 80, 90]
}
gridSearch = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv = 5)
grid_result = gridSearch.fit(X_train, y_train)

print(gridSearch.best_score_)
print(gridSearch.best_params_)

scores = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']


depth = grid_result.cv_results_['param_max_depth']
min_sample = grid_result.cv_results_['param_min_samples_split']



sns.lineplot(x =  min_sample, y= scores, hue = depth)
plt.title('GridSearch for Depth and Minimum Sample Split (CV)')
plt.xlabel('Minimum Leaf Sample for Splits')
plt.ylabel('CV Score')
plt.legend(title = 'Tree Depth')

# Not to big of a difference between tree depths, highest train accuracy is obtained with smaller minimum sample, but really we care about test accuracy so will iterate over those to see what our test accuracy looks like
# check to see if test error varies from the cross-validation test error estimates
depth_list = [5,10,15]
min_sample_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
depths = []
min_samples = []
scores = []
for depth in depth_list:
    for min_sample in min_sample_list:
        fullClassTree = DecisionTreeClassifier(max_depth = depth, min_samples_leaf = min_sample, random_state = 42)
        fullClassTree.fit(X_train, y_train)
        y_predicted = fullClassTree.predict(X_test)
        depths = depths + [depth]
        min_samples = min_samples + [min_sample]
        scores = scores + [fullClassTree.score(X_test, y_test)]


sns.lineplot(x = min_samples, y = scores, hue = depths)
plt.title('Test Score for different Depth and Minimum Sample Split (Test Data)')
plt.xlabel('Minimum Leaf Sample for Splits')
plt.ylabel('Test Score')
plt.legend(title = 'Tree Depth')

# no limit on number of min_samples to see what we get, accuracy not too too much better
fullClassTree_v2 = DecisionTreeClassifier(random_state = 42)
fullClassTree_v2.fit(X_train, y_train)
y_predicted = fullClassTree_v2.predict(X_test)
fullClassTree_v2.score(X_test, y_test)

export_graphviz(fullClassTree_v2, out_file = 'fullClassTree_v2.dot', feature_names = X_train.columns)

classificationSummary(y_train, fullClassTree_v2.predict(X_train))

X1 = ebay_df_final.drop(['ClosePrice', 'Competitive?'], axis = 1)
y1 = ebay_df_final['Competitive?']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.4, random_state = 42)

fullClassTree_noClosePrice = DecisionTreeClassifier(min_samples_leaf = 50, random_state = 42)
fullClassTree_noClosePrice.fit(X1_train, y1_train)
y_predicted = fullClassTree_noClosePrice.predict(X1_test)
fullClassTree_noClosePrice.score(X1_test, y1_test)

# training data confusion matrix
classificationSummary(y1_train, fullClassTree_noClosePrice.predict(X1_train))

# testing data confusion matrix
classificationSummary(y1_test, fullClassTree_noClosePrice.predict(X1_test))

# performing a grid search
param_grid = {
    'max_depth' : [5,10,15],
    'min_samples_split':[10, 20, 30, 40, 50, 60, 70, 80, 90]
}
gridSearch = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv = 5)
grid_result = gridSearch.fit(X1_train, y1_train)
scores = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']
depth = grid_result.cv_results_['param_max_depth']
min_sample = grid_result.cv_results_['param_min_samples_split']

sns.lineplot(x =  min_sample, y= scores, hue = depth)
plt.title('GridSearch for Depth and Minimum Sample Split (CV)')
plt.xlabel('Minimum Leaf Sample for Splits')
plt.ylabel('CV Score')
plt.legend(title = 'Tree Depth')

# check to see if test error varies from the cross-validation test error estimates
depth_list = [5,10,15]
min_sample_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
depths = []
min_samples = []
scores = []
for depth in depth_list:
    for min_sample in min_sample_list:
        fullClassTree = DecisionTreeClassifier(max_depth = depth, min_samples_leaf = min_sample, random_state = 42)
        fullClassTree.fit(X1_train, y1_train)
        y_predicted = fullClassTree.predict(X1_test)
        depths = depths + [depth]
        min_samples = min_samples + [min_sample]
        scores = scores + [fullClassTree.score(X1_test, y1_test)]

sns.lineplot(x = min_samples, y = scores, hue = depths)
plt.title('Test Score for different Depth and Minimum Sample Split (Test Data)')
plt.xlabel('Minimum Leaf Sample for Splits')
plt.ylabel('Test Score')
plt.legend(title = 'Tree Depth')

export_graphviz(fullClassTree_noClosePrice, out_file = 'fullClassTree_noClosePrice.dot', feature_names = X1_train.columns)


bestClassTree = gridSearch.best_estimator_

plt.figure(figsize=(25,20))
tree.plot_tree(bestClassTree, feature_names = X1.columns, class_names = ['NC', 'C'], filled = True)

# Open fullClassTree_noClosePrice.dot and copy the text into http://webgraphviz.com/ to visualize the tree

# create a scatterplot for two best quantitative predictors
fig, ax = plt.subplots(figsize = ( 15 , 10 ))
ax.set_xlim(-5,150)  # ranges from 0 to 1000, however majority are fewer than 200, so will limit for visualizing
ax.set_ylim(-5, 15000)  # ranges from 0 to 35000+, put majority are less than 15000 so will limit for visualizing
sns.scatterplot(ebay_df_final['OpenPrice'], ebay_df_final['sellerRating'], hue = ebay_df_final['Competitive?'])

# Performing decision tree with selected predictors
pd.options.display.max_columns = None  # Do not hide column info
pd.options.display.max_rows = None  # Do not hide row info
dataset = pd.read_excel("ebayAuctions.xlsx", sheet_name="eBay auctions")

# Data Cleaning
dataset.info()
dataset.rename(columns={"Competitive?": "Competitive"}, inplace=True)
dataset["Competitive"] = dataset["Competitive"].astype("string")  # Change Competitive to categorical
dataset["Duration"] = dataset["Duration"].astype("string")  # Change Duration to categorical
for columnName in dataset.columns:  # Change all object type to string type
    if dataset[columnName].dtype == "object":
        dataset[columnName] = dataset[columnName].astype("string")
duplication = ~dataset.duplicated()  # There are a lot of duplicated errors. ~ to invert pandas boolean series
cleanedData = dataset
cleanedData.reset_index(drop=True, inplace=True)
cleanedData.info()

# Data preprocessing
cateVariable = []  # Create a list of feature names that are categorical for one hot encoding
for columnName in cleanedData.columns:
    if cleanedData[columnName].dtype == "string":
        cateVariable.append(columnName)
cateVariable.remove("Competitive")  # We do not want to change target variable into dummies
cleanedData_dummy = pd.get_dummies(data=cleanedData, columns=cateVariable)  # Create dummy variables
data_x = cleanedData_dummy.drop("Competitive", axis="columns")
data_y = cleanedData_dummy["Competitive"]
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=0.6, random_state=1)

# Classification Tree with terminal node >= 50
tree_model = DecisionTreeClassifier(min_samples_leaf=50, random_state=1)
tree_model.fit(x_train, y_train)
prediction = tree_model.predict(x_test)
confusionMatrix1 = confusion_matrix(y_test, prediction, labels=["1", "0"])
model1Accuracy = accuracy_score(y_test, prediction)
print("Model 1's accuracy is: " + str(model1Accuracy))
# Plot the tree
plt.figure(figsize=(20, 20))
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
tree.plot_tree(tree_model, fontsize=10)
plt.show()
# Feature importance
importance_model1 = tree_model.feature_importances_
importance_table = pd.DataFrame(columns=["Feature", "Importance"])  # Create an importance table to plot bar chart
featureNum = 0
for score in importance_model1:
    print("feature " + str(featureNum) + "'s importance score: " + str(score) + " (" + x_train.columns[featureNum] + ")")
    rowAdded = pd.DataFrame([[x_train.columns[featureNum], score]], columns=["Feature", "Importance"])
    importance_table = pd.concat([importance_table, rowAdded])
    featureNum = featureNum + 1

# Plot a bar chart to visualize feature importance
plt.figure(figsize=(10, 14))
sns.barplot(data=importance_table, x="Feature", y="Importance")
plt.title("Feature Importance")
plt.subplots_adjust(bottom=0.2, top=0.95)
plt.xticks(rotation=90)
plt.show()
# Important features are:
# Feature 0: SellerRating
# Feature 1: ClosePrice
# Feature 2: OpenPrice
# Feature 4: Category_Automotive
# Feature 7: Category_Clothing/Accessories
# Feature 15: Category_Jewelry
# Feature 20: Category_Toys/Hobbies
# Feature 21: Currency_EUR


# Classification Tree with unimportant features removed
data_reducedFeature = cleanedData[["Category", "Currency", "sellerRating", "OpenPrice", "Competitive"]]
data_reducedFeature_dummy = pd.get_dummies(data=data_reducedFeature, columns=["Category", "Currency"])
data_x = data_reducedFeature_dummy.drop("Competitive", axis="columns")
data_y = data_reducedFeature_dummy["Competitive"]
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=0.6, random_state=1)
tree_model2 = DecisionTreeClassifier(min_samples_split=50, random_state=1)
tree_model2.fit(x_train, y_train)
prediction2 = tree_model2.predict(x_test)
confusionMatrix2 = confusion_matrix(y_test, prediction2, labels=["1", "0"])
model2Accuracy = accuracy_score(y_test, prediction2)
print("Model 2's accuracy is: " + str(model2Accuracy))
print(confusionMatrix2)

# Plot the new tree
plt.figure(figsize=(20, 20))
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
tree.plot_tree(tree_model2, fontsize=10)
plt.show()

# Most important features are ClosePrice and OpenPrice
# Plot the classification graph
sns.scatterplot(data=cleanedData, x="OpenPrice", y="ClosePrice", hue="Competitive")
plt.show()




