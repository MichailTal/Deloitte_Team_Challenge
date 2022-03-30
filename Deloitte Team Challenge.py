import os
import pandas as pd

## 0. Loading the Data
path = os.path.join('datasets', 'Deloitte Team Challenge')
data_path = os.path.join(path, 'chair_automization.csv')

chair_data = pd.read_csv(data_path)

# print(str(chair_data))  ## checks for correct data read


## 1. Splitting into train and test data


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(chair_data[['Gender (1=Female 0=Male)', 'Apparent '
                                                                                            'Temperature ('
                                                                                            'C)',
                                                                'Humidity', 'Wind Speed (km/h)', 'Visibility (km)',
                                                                'Body pain', 'Daily hours spent in chair', 'Height',
                                                                'Weight', 'Shoulder-/Arm length', 'Height (sitting)',
                                                                'Height (standing)', 'Ergonomic chair']],
                                                    chair_data[['Preference for '
                                                                'relaxation '
                                                                'settings']],
                                                    test_size=0.2, random_state=42)

num_attribs = ['Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Visibility (km)',
               'Daily hours spent in chair', 'Height', 'Weight', 'Shoulder-/Arm length', 'Height (sitting)',
               'Height (standing)']

cat_attribs = ['Gender (1=Female 0=Male)', 'Body pain', 'Ergonomic chair']

## 2. Building a pipeline

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

from sklearn.preprocessing import OneHotEncoder

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('cat_encoder', OneHotEncoder(sparse=False)),
])

from sklearn.compose import ColumnTransformer

final_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs),
])

X_train = final_pipeline.fit_transform(chair_data[num_attribs + cat_attribs])
y_train = chair_data[['Preference for relaxation settings']]

## Introducing the RandomForestClassifier


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]

forest_reg = RandomForestClassifier()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train.values.ravel())

# print(grid_search.best_params_)

# print(grid_search.best_estimator_)

forest_clf = RandomForestClassifier(max_features=4, n_estimators=30, random_state=42)
forest_clf.fit(X_train, y_train.values.ravel())
forest_scores = cross_val_score(forest_clf, X_train, y_train.values.ravel(), cv=10)

print(forest_scores.mean())  ## Prints out the accuracy given the setting
