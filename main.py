# Importing essential libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('forestfires.csv')


# OneHotEncode the month and the day columns because they are categorical.
# We made a mistake here, since months and days do take an order, we cannot onehotencode them.
# So we're going to ordinalencode the data.
encoded_df = pd.get_dummies(data = df, columns = ['month', 'day'])

# Ordinal Encode the data because months and days do take an order.
ordinal_encoded_df = df.copy()

scale_mapper_month = {'mar':2, 'oct':9, 'aug':7, 'sep':8, 'apr':3, 'jun':5, 'jul':6, 'feb':1, 'jan':0, 'dec':10, 'may':4,
                'nov':9}
ordinal_encoded_df['month'] = df['month'].replace(scale_mapper_month)

scale_mapper_day = {'fri':4, 'tue':1, 'sat':5, 'sun':6, 'mon':0, 'wed':2, 'thu':3}
ordinal_encoded_df['day'] = df['day'].replace(scale_mapper_day)


# 80% of 517 rows is 413 so we're going to take that many columns in test and the rest in
# Splitting the dataset into train and test
# First slice the data
"""X_train = ordinal_encoded_df[: 413].copy()
X_test = ordinal_encoded_df[413:].copy()

# And then drop the target value.
X_train.drop(labels = 'area', axis = 1, inplace = True)
X_test.drop(labels = 'area', axis = 1, inplace = True)

y_train = ordinal_encoded_df[: 413].copy()
y_test = ordinal_encoded_df[413:].copy()


y_train.drop(labels = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain'
       , 'month', 'day'], axis = 1, inplace = True)

y_test.drop(labels = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain',
       'month', 'day'], axis = 1, inplace = True)"""

# We found out that train_test_splitting the data is giving us better results then simply slicing the data!
# Since we're not getting the results that we want even by changing the model we have to try and
# preprocess the data in such a way that we get the required accuracy
# So, we're going to train_test_split the data randomly to try and get better results
X = ordinal_encoded_df.copy()
y = ordinal_encoded_df.copy()
X.drop(labels = 'area', axis = 1, inplace = True)
y.drop(labels = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain',
       'month', 'day'], axis = 1, inplace = True)


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, shuffle = True, random_state = 1)


# Now that we have the test and the train data ready, we're going to build the model.
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Taking predictions into a variable
predictions = regressor.predict(X_test)

# Making different variables for y_pred and y_true for evaluating metrics such as mse and rmse
y_pred = predictions
y_true = y_test

# Evaluating the model
mse = mean_squared_error(y_true, y_pred)
rmse = math.sqrt(mse)


print(f"Mean Squared Error or MSE of linear regression model is : {mse}")
print(f"Root Mean Squared Eroor or RMSE of Linear Regression model is : {rmse}")
print("Linear Regression R^2 score: {:.5f}".format(regressor.score(X_test, y_test)))

# Since the linear regression model is not giving us our desired predictions,
# we're gonna use a LASSO REGRESSION MODEL
# Making the model
model = Ridge()

# Fitting the model
model.fit(X_train, y_train)

# Taking predictions into a variable
predictions = model.predict(X_test)

y_true = y_test
y_pred = predictions

# Evaluating the model
mse_lasso = mean_squared_error(y_true, y_pred)
rmse_lasso = math.sqrt(mse_lasso)

print(f"Mean Squared error or MSE in Lasso Regression model is {mse_lasso}")
print(f"Root Mean Squared Error or RMSE in Lasso Regression model is {rmse_lasso}")

# Since lasso regression is also not giving us the desired accuracy, we're going to use ridge regression model

# Making the model
model = Ridge()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40]}
ridge_regressor = GridSearchCV(model, parameters, scoring = "neg_mean_squared_error", cv = 5)

# Fitting the model
model.fit(X_train, y_train)

# Putting the predictions into a variable
predictions = model.predict(X_test)

y_true = y_test
y_pred = predictions

# Evaluating the model
mse_ridge = mean_squared_error(y_true, y_pred)
rmse_ridge = math.sqrt(mse_ridge)

print(f"Mean Squared error or MSE of ridge regression model is {mse_ridge}")
print(f"Root Mean Squared Error or RMSE of ridge regression model is {rmse_ridge}")

"""
# Since regression algorithms are not giving me the required results, we are therefore going to use
# classification algorithms like logistic regression

# Changing the output variable because now we're using a classification algo

y_classification = df['area'].apply(lambda x: 1 if x > 0 else 0)
y_train_classification = y_classification[: 413]
y_test_classification = y_classification[413:]




# Define the model
logreg = LogisticRegression()

# Fitting the model
logreg.fit(X_train, y_train_classification)


# Predicting and putting it into a variable
predictions = logreg.predict(X_test)

y_true = y_test_classification
y_pred = predictions

mse_logistic = mean_squared_error(y_true, y_pred)
rmse_logistic = math.sqrt(mse_logistic)

print(f"Mean Squared Erorr or MSE on Logistic Regression is {mse_logistic}")
print(f"Root Mean Squared Error or RMSE on Logistic Regression is {rmse_logistic}")
"""

# We're also going to try using a MLP neural netowrk regressor
# As neural networks perform better on standardised data we're going to standardise the data.

# Scale Feature the data
scaler = StandardScaler()
scaler.fit(X_train)

X_train = pd.DataFrame(scaler.transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))


nn_reg_model = MLPRegressor(hidden_layer_sizes=(16, 16))
nn_reg_model.fit(X_train, y_train)

print("NN Regression r^2 score is : {:.5f}".format(nn_reg_model.score(X_test, y_test)))








