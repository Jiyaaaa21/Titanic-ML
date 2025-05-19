# import required libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Dataset

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(train_df.shape)
print(train_df.columns)
print(train_df.head(3))

print(train_df.isnull().sum()) # check for null values

print(train_df.describe(include='all')) 

sns.countplot(data=train_df, x='Survived', hue='Sex')
plt.title('Survival Count by Sex')
plt.show()

sns.countplot(data=train_df, x='Survived', hue='Pclass')
plt.title('Survival Count by Passenger Class')
plt.show()

test_df['Survived'] = -1  
combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)

combined['Age'] = combined.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
combined['Embarked'] = combined['Embarked'].fillna(combined['Embarked'].mode()[0])
combined['Fare'] = combined['Fare'].fillna(combined['Fare'].median())
combined['Cabin'] = combined['Cabin'].fillna('Missing')

print(train_df.isnull().sum())


combined['Title'] = combined['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False) # To extract titles from name

combined['Title'] = combined['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',                                               'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
combined['Title'] = combined['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

# combined.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1, inplace=True) # Dropping unnecessary cloumns

# combined = pd.get_dummies(combined, columns=['Sex', 'Embarked', 'Title'], drop_first=True)

# # To Restore train and test sets
# train_processed = combined[combined['Survived'] != -1]
# test_processed = combined[combined['Survived'] == -1].drop(['Survived'], axis=1)

# X = train_processed.drop('Survived', axis=1)
# y = train_processed['Survived']

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report


# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) # Split into train and test set


# rf = RandomForestClassifier(n_estimators=100, random_state=42) # Random Forest Model
# rf.fit(X_train, y_train)

# # Predict and evaluate
# y_pred = rf.predict(X_val)
# print("Accuracy:", accuracy_score(y_val, y_pred))
# print(classification_report(y_val, y_pred))


# # To predict test set results.

# test_predictions = rf.predict(test_processed)

# original_test = pd.read_csv('/mnt/data/titanic_data/test.csv')
# submission = pd.DataFrame({
#     'PassengerId': original_test['PassengerId'],
#     'Survived': test_predictions
# })

# submission.to_csv('/mnt/data/titanic_submission.csv', index=False)
# print("Submission file saved as titanic_submission.csv")

