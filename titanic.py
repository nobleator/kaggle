"""Kaggle Titanic Problem

"""
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 08:48:09 2018

@author: bnoble
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


# Read in training and testing data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
"""
Review Pandas methods, such as:
    train_df.head()
    train_df.tail()
    train_df.info()
    train_df.describe()
    pd.crosstab()

Check for survival rate (in this case, for Pclass):
train_df[['Pclass', 'Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
"""
### Data Cleaning
# Remove ticket and cabin categories
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
guess_ages = np.zeros((2,3))
for dataset in combine:
    # Create new category "Title" based off of the name category
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # TODO: For more robust analysis, do a count on titles and replace those under a certain percentage
    # Replace and condense titles and convert to numerical
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    # Replace categorical sex with numerical sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    # Replace missing ages with estimated values based on correlations
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)
    # TODO: Use actual ages for prediction instead of bands?
    # Convert age into bands
    dataset['AgeBand'] = pd.cut(dataset['Age'], 5)
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
    # Create new categories for family size and singleness
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    # Create new category for age times class
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
    # Fill in missing embarkation data and replace with numerical category
    freq_port = dataset.Embarked.dropna().mode()[0]
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    # Replace missing fare data
    dataset['Fare'].fillna(dataset['Fare'].dropna().median(), inplace=True)
    # Create fare bands and convert to ordinals
    dataset['FareBand'] = pd.qcut(dataset['Fare'], 4)
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

# Remove passengerId (arbitrary category)
# Remove name in favor of title
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
# Remove age band (was only a placeholder)
train_df = train_df.drop(['AgeBand'], axis=1)
test_df = test_df.drop(['AgeBand'], axis=1)
# Remove Parch, SibSp, and FamilySize in favor of IsAlone
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
#Remove fare band (was only a placeholder)
train_df = train_df.drop(['FareBand'], axis=1)
test_df = test_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

### Model Construction
X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis=1).copy()
# Logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_logreg = round(logreg.score(X_train, Y_train) * 100, 4)
print('_' * 80)
print('Logistic regression accuracy: {0}% fit'.format(acc_logreg))
# Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, Y_train)
Y_pred = lasso.predict(X_test)
acc_lasso = round(lasso.score(X_train, Y_train) * 100, 4)
print('_' * 80)
print('Lasso accuracy: {0}% fit'.format(acc_lasso))
# Ridge Regression
ridgereg = Ridge(fit_intercept=True, alpha=0.1)
ridgereg.fit(X_train, Y_train)
Y_pred = ridgereg.predict(X_test)
acc_ridgereg = round(ridgereg.score(X_train, Y_train) * 100, 4)
print('_' * 80)
print('Ridge regression accuracy: {0}% fit'.format(acc_ridgereg))
submission = pd.DataFrame({"PassengerId": test_df["PassengerId"],
                           "Survived": logreg.predict(X_test)})
submission.to_csv('submission.csv', index=False)
