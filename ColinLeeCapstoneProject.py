#%% imports
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from scipy.stats import mannwhitneyu
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import spearmanr

#%% Pre-processing
nNumber = 15477588
# name the columns
col1 = ['meanRating', 'meanDifficulty', 'numRatings', 'pepper','takeAgainPercentage', 
        'numRatingsOnline', 'male', 'female']
col2 = ['field', 'university', 'state']

#import data and assign the column names
num = pd.read_csv('/Users/colin/Desktop/Capstone Project/rmpCapstoneNum.csv', header = None)
num.columns = col1

qual = pd.read_csv('/Users/colin/Desktop/Capstone Project/rmpCapstoneQual.csv', header = None)
qual.columns = col2

# join the dfs to make 1 df with everything.
df = num.join(qual)
df = df[df['numRatings'] >= 15]
# There are 4679 different profs with >= 15 ratings. We will only consider those profs
# for the entirety of this assignment

#%%
# Task 1:
# alpha = 0.005
# for this task, we will also get rid of rows where both Male and Female are the same value.
# since those profs are neither definitively males or females.

# Null Hyp: Male = Female
# Alt Hyp: Male > Female
df1 = df[(df['male'] != df['female']) & (df['male'].isin([0,1]))].copy()
df1 = df1.dropna()

males = df1[df1['male'] == 1]
females = df1[df1['female'] == 1]

# visualization
plt.hist(males['meanRating'], bins=20, alpha=0.6, label='Male', edgecolor='black')
plt.hist(females['meanRating'], bins=20, alpha=0.6, label='Female', edgecolor='black')
plt.xlabel('Mean Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Mean Ratings by Gender')
plt.legend()
plt.tight_layout()
plt.show()

# investigate any large r's for possible confounding variables
corr_matrix = df1.select_dtypes(include=[np.number]).corr()

# visualization
plt.hist(males['meanDifficulty'], bins=20, alpha=0.6, label='Male', edgecolor='black')
plt.hist(females['meanDifficulty'], bins=20, alpha=0.6, label='Female', edgecolor='black')
plt.xlabel('Mean Difficulty')
plt.ylabel('Frequency')
plt.title('Distribution of Mean Difficulty by Gender')
plt.legend()
plt.tight_layout()
plt.show()


# sig test on meanDifficulty in male vs female taught classes to see if
# males teach a different difficulty of classes.
statDiff, p_valueDiff = ttest_ind(males['meanDifficulty'], females['meanDifficulty'])

# proportions z-test to see if more males or females received pepper
count_pepper_male = df1[df1['male'] == 1]['pepper'].sum()
n_male = (df1['male'] == 1).sum()

count_pepper_female = df1[df1['male'] == 0]['pepper'].sum()
n_female = (df1['male'] == 0).sum()

count = [count_pepper_male, count_pepper_female]
nobs = [n_male, n_female]

statPepper, p_valuePepper = proportions_ztest(count, nobs, alternative='two-sided')

### ACTUAL ANSWER
stat, p_value = mannwhitneyu(males['meanRating'], females['meanRating'], alternative='greater')
#%%
# Task 2:
# alpha = 0.005
df2 = df[['meanRating', 'numRatings']].copy().dropna()

# Divide into 2 groups: greater than median and less than median
numRatingsMedian = df2.numRatings.median()
df2AboveMedian = df2[df2['numRatings'] >= numRatingsMedian]
df2BelowMedian = df2[df2['numRatings'] < numRatingsMedian]

# visualization
plt.hist(df2AboveMedian['meanRating'], bins=20, alpha=0.6, label='Above Median', edgecolor='black')
plt.hist(df2BelowMedian['meanRating'], bins=20, alpha=0.6, label='Below Median', edgecolor='black')
plt.xlabel('Mean Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Mean Ratings by Number of Ratings')
plt.legend()
plt.tight_layout()
plt.show()

# U-test
stat2, p_value2 = mannwhitneyu(df2AboveMedian['meanRating'], df2BelowMedian['meanRating'], alternative='two-sided')


#%%
# Task 3
# relationship between meanRating and meanDifficulty
df3 = df[['meanRating', 'meanDifficulty']].copy()

#Pearson Corr
corrs3 = df3.corr()
r = corrs3['meanRating'].iloc[1]

# Spearman corr
spearman_corr, pvalue = spearmanr(df3['meanRating'], df3['meanDifficulty'])

# moderately strong, negative relationship between meanRating and meanDifficulty

# Simple linear regression model
X = df3[['meanDifficulty']]
y = df3['meanRating']
reg = LinearRegression()
reg.fit(X,y)
print(reg.coef_)

# visualization
plt.scatter(df3['meanDifficulty'], df3['meanRating'], alpha=0.3, label='Data')

x_vals = np.linspace(df3['meanDifficulty'].min(), df3['meanDifficulty'].max(), 100)
y_vals = reg.predict(x_vals.reshape(-1, 1))

plt.plot(x_vals, y_vals, color='red', label='Regression line')
plt.xlabel('Mean Difficulty')
plt.ylabel('Mean Rating')
plt.title('Mean Rating vs Mean Difficulty')
plt.legend()
plt.show()
# for every 1 point increase in meanDifficulty, the expected meanRating decreases by 0.760
#%%
# Task 4
df4 = df[['meanRating','numRatings','numRatingsOnline']].copy()

df4['percentageOnline'] = df4['numRatingsOnline'] / df4['numRatings']

# Split in to to groups: above 40% and below 40%

df4Above40 = df4[df4['percentageOnline'] >= .40]
df4Below40 = df4[df4['percentageOnline'] < .40]

# visualization
plt.hist(df4['meanRating'], bins=20, alpha=0.6, edgecolor='black')
plt.xlabel('Mean Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Mean Ratings by Percentage of Ratings that are Online')
plt.tight_layout()
plt.show()

# U-test
stat4, p_value4 = mannwhitneyu(df4Above40['meanRating'], df4Below40['meanRating'], alternative='two-sided')


#%%
# Task 5
# relationship between meanRating and takeAgainPercentage
df5 = df[['meanRating', 'takeAgainPercentage']].copy().dropna()

#pearson corr
corrs = df5.corr()
r = corrs['meanRating'].iloc[1]

# spearman corr
spearman_corr, pvalue = spearmanr(df5['meanRating'], df5['takeAgainPercentage'])

# very strong, positive relationship between meanRating and takeAgainPercentage
# simple linear regression
X = df5[['takeAgainPercentage']]
y = df5['meanRating']
reg = LinearRegression()
reg.fit(X,y)
print(reg.coef_)

# visualization
plt.figure(figsize=(8, 6))
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, reg.predict(X), color='red', label='Regression Line')
plt.xlabel('Take Again Percentage')
plt.ylabel('Mean Rating')
plt.title('Mean Rating vs Take Again Percentage')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# for every 1 point increase in takeAgainPercentage, the expected meanRating increases by 0.0307

#%%
# Task 6
df6 = df[['meanRating','pepper']].copy()

# split data into 2 groups: hot and not hot
df6Hot = df6[df6['pepper'] == 1]
df6NotHot = df6[df6['pepper'] == 0]

# visualization
plt.hist(df6Hot['meanRating'], bins=20, alpha=0.6, label='Hot', edgecolor='black')
plt.hist(df6NotHot['meanRating'], bins=20, alpha=0.6, label='Not Hot', edgecolor='black')
plt.xlabel('Mean Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Mean Ratings by Appearance')
plt.legend()
plt.tight_layout()
plt.show()

# U-test
stat6, p_value6 = mannwhitneyu(df6Hot['meanRating'], df6NotHot['meanRating'], alternative='greater')

#%%
# Task 7
# Simple linear regression
lin = LinearRegression()

X = df[['meanDifficulty']]
y = df['meanRating']

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state = nNumber)

# train model
lin.fit(X_train, y_train)

# test model
linear_pred = lin.predict(X_test)

# calculate stats
lin_rmse = root_mean_squared_error(y_test, linear_pred)
linR2 = lin.score(X_test, y_test)

# visualization
plt.scatter(df['meanDifficulty'], df['meanRating'])
plt.xlabel('Mean Difficulty')
plt.ylabel('Mean Rating')
plt.title('Scatter Plot of Difficulty vs Rating')
x_vals = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_vals = lin.predict(x_vals)
plt.plot(x_vals, y_vals, color='red', label='Regression Line')
plt.show()
#%%
# Task 8
# multiple regression
multiLin = LinearRegression()
cols = ['meanDifficulty', 'numRatings', 'pepper','takeAgainPercentage', 
        'numRatingsOnline', 'male', 'female', 'meanRating']
df_clean = df[cols].dropna()

X = df_clean.drop('meanRating', axis=1)
y = df_clean['meanRating']

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=nNumber)
multiLin = LinearRegression()

#train model
multiLin.fit(X_train, y_train)

#test model
y_pred = multiLin.predict(X_test)

multiRMSE = root_mean_squared_error(y_test, y_pred)
multiR2 = multiLin.score(X_test, y_test)

residuals = y_test - y_pred

# visualization

plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Mean Rating")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Mean Rating")
plt.ylabel("Predicted Mean Rating")
plt.title("Actual vs Predicted")
plt.show()
#%%
# Task 9
# Logistic regression with 1 predictor.
df9 = df[['meanRating', 'pepper']].dropna()

X = df9[['meanRating']]
y = df9['pepper']
print(df9.pepper.value_counts())

# Split data into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=nNumber)


# train model
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# test model
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_proba)

X_sorted = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_logit = model.predict_proba(X_sorted)[:, 1]

# visualization
plt.figure(figsize=(8, 6))
plt.scatter(X_train[y_train == 0], y_train[y_train == 0], color='blue', label='No Pepper', alpha=0.6)
plt.scatter(X_train[y_train == 1], y_train[y_train == 1], color='red', label='Pepper', alpha=0.6)
plt.plot(X_sorted, y_logit, color='black', linewidth=2, label='Logistic Regression')
plt.xlabel('Mean Rating')
plt.ylabel('Probability of Pepper')
plt.title('Logistic Regression: Predicting Pepper from Mean Rating')
plt.legend()
plt.grid(True)
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Predicting Pepper from Mean Rating')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
# Task 10

features = ['meanDifficulty', 'numRatings', 'meanRating',
            'takeAgainPercentage', 'numRatingsOnline', 'male', 'female']
target = 'pepper'

df_model = df[features + [target]].dropna()

X = df_model[features]
y = df_model[target]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Train model
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# test model
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]


auc_full = roc_auc_score(y_test, y_proba)

# visualization
# Step 6: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'All Features (AUC = {auc_full:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Pepper Prediction')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient')

plt.figure(figsize=(8, 6))
plt.barh(coef_df['Feature'], coef_df['Coefficient'])
plt.axvline(0, color='black', linestyle='--')
plt.xlabel('Coefficient Value')
plt.title('Logistic Regression Coefficients: Pepper Prediction')
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
# Extra Credit Task

# i am interested to see if california and texas profs are rating differently.
ca = df[df['state'] == 'CA']
tx = df[df['state'] == 'TX']

# visualization
plt.hist(ca['meanRating'], bins=20, alpha=0.6, label='California', edgecolor='black')
plt.hist(tx['meanRating'], bins=20, alpha=0.6, label='Texas', edgecolor='black')
plt.xlabel('Mean Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Mean Ratings by State')
plt.legend()
plt.tight_layout()
plt.show()

# u-test
statState, p_valueState = mannwhitneyu(ca['meanRating'], tx['meanRating'], alternative='two-sided')