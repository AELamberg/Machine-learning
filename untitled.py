import numpy as np                   # import numpy package under shorthand "np"
import pandas as pd                  # import pandas package under shorthand "pd"
import matplotlib.pyplot as plt


# Regression import 

%config Completer.use_jedi = False  # enable code auto-completion
from sklearn.preprocessing import PolynomialFeatures    # function to generate polynomial and interaction features
from sklearn.linear_model import LinearRegression    # classes providing Linear Regression with ordinary squared error loss and Huber loss, respectively
from sklearn.metrics import mean_squared_error    # function to calculate mean squared error 
from sklearn.tree import DecisionTreeClassifier

training_df = pd.read_csv('LCK_data.csv')     # read the data from file, initiate dataframe
training_df = training_df.drop(['Region', 'Season', 'Games', 'Game duration'], axis=1)    # drop data which doesn't contribute to result of a game
training_df = training_df.drop(['NASHPG', 'TD@15', 'DPM', 'WPM', 'VWPM', 'WCPM', 'PPG', 'HERPG', 'DRAPG', 'DRA@15'], axis=1)   # drop some data to limit the amount of features
validation_df = training_df.drop([1, 2, 3, 5, 6, 9, 0], axis=0)    # create the validation dataframe, drop the teams used in the training dataframe
training_df = training_df.drop([4, 7, 8], axis=0)    # drop the validation data from training dataframe
#display(training_df)   # display the data tables
#display(validation_df)

training_statistics = training_df.drop(['Name', 'Win rate'], axis=1)    # this code block initializes and checks the training/validation data for use
tr_teams = training_df['Name'].to_numpy()
validation_statistics = validation_df.drop(['Name', 'Win rate'], axis=1)
win_rates = training_df['Win rate'].to_numpy()
validation_wrs = validation_df['Win rate'].to_numpy()   # create the validation labels

fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))
yticks = np.linspace(0, 1, 11)
degrees = []
errors = []


for i in range (1, 8):
    poly = PolynomialFeatures(degree=i)    # initialize polynomial features class, fit the data in polynomial functions
    polynomial_data = poly.fit_transform(training_statistics)

    poly_regr = LinearRegression(fit_intercept=False)
    poly_regr.fit(polynomial_data, win_rates)

    poly_stats = poly.fit_transform(validation_statistics)    # polynomial prediction on validation set 
    poly_pred = poly_regr.predict(poly_stats)
    val_poly_error = mean_squared_error(validation_wrs, poly_pred)
    
    axes1[0].scatter(['Hanhwa Life eSports', 'Nongshim RedForce', 'OK BRION'], poly_pred, label = f'degree = {i}', marker = 'o', linestyle = 'None') # this block plots the predictions
    degrees.append(f'degree={i}')
    errors.append(val_poly_error)
    axes1[1].scatter(f'degrees={i}', val_poly_error, marker = 'o', linestyle = 'None')
    axes1[1].annotate(f'{round(val_poly_error, 7)}', (f'degrees={i}',val_poly_error), textcoords='offset points', xytext = (-5, 5), ha= 'center')

    
        
# this block plots the real win rates and cleans the plots 
axes1[0].scatter(['Hanhwa Life eSports', 'Nongshim RedForce', 'OK BRION'], validation_wrs, label = 'True win rates', marker = 'o', linestyle = 'None') 
axes1[0].legend()  
axes1[0].set_yticks(yticks)
axes1[0].set_title('Polynomial regression validation set predicted win rates with degrees 1-7')
axes1[1].set_title('Polynomial regression validation mean squared error with degrees 1-7')

pred_stats = pd.read_csv('LCK_Spring_2024_stats.csv')    # initialize test set and labels, display the table
test_wrs = pred_stats['Win rate'].to_numpy()
test_teams = pred_stats['Name'].to_numpy()
pred_stats = pred_stats.drop(['Region', 'Season', 'Games', 'Game duration', 'Win rate', 'NASHPG', 'TD@15', 'DPM', 'WPM', 'VWPM', 'WCPM', 'PPG', 'HERPG', 'DRAPG', 'DRA@15'], axis=1) # standardization
display(pred_stats)
pred_stats = pred_stats.drop(['Name'], axis=1)

poly_test = PolynomialFeatures(degree=3)    # initialize polynomial features and regression model
poly_test_regr = LinearRegression(fit_intercept=False)
poly_test_regr.fit(polynomial_data, win_rates)

test_stats = poly.fit_transform(pred_stats)    # make the prediction with the model and calculate MSE
test_pred = poly_test_regr.predict(test_stats)
test_poly_error = mean_squared_error(test_wrs, test_pred)

plt.figure(figsize=(18, 6))    # visualize results on a plot
plt.plot(test_teams, test_pred, marker = 'o', linestyle = 'None', label = 'LCK Spring 2024 predicted win rates')
plt.plot(test_teams, test_wrs, marker = 'o', linestyle = 'None', label = 'LCK Spring 2024 true win rates')
plt.yticks(yticks)
plt.title(f'LCK Spring 2024 predicted win rates vs. true win rates\nPolynomial regression, third degree\nMSE:{test_poly_error}')
plt.legend()

fig, axes = plt.subplots(2, 2, figsize=(22, 10))
yticks = np.linspace(0, 1, 11)
yticks100 = np.linspace(0, 100, 11)
yticks_error = np.linspace(0, 500, 11)
depths = []
errors = []
for i in range(0, 10):   # round the labels into categories and into full percentages
    if i < 7:
        win_rates[i] = round(win_rates[i], 1) * 100
    if i < 3:
        validation_wrs[i] = round(validation_wrs[i], 1) * 100
    test_wrs[i] = round(test_wrs[i], 1) * 100
    
print(win_rates)    
    
for i in range (1, 11):
    dte = DecisionTreeClassifier(max_depth=i)    # initialize DTC class, fit the model
    dte.fit(training_statistics, win_rates)

    validation_wrs = validation_df['Win rate'].to_numpy()   # create the validation labels

    dte_tr_pred = dte.predict(training_statistics)    # DTE prediction on validation set
    dte_val_pred = dte.predict(validation_statistics)
    tr_dte_error = mean_squared_error(win_rates, dte_tr_pred)
    val_dte_error = mean_squared_error(validation_wrs, dte_val_pred)

    axes[0][0].scatter(['Hanhwa Life eSports', 'Nongshim RedForce', 'OK BRION'], dte_val_pred, label = f'maxdepth = {i}', marker = 'o', linestyle = 'None') # this block plots the predictions
    axes[1][0].scatter(tr_teams, dte_tr_pred, label = f'maxdepth = {i}', marker = 'o', linestyle = 'None')
    depths.append(f'maxdepth={i}')
    errors.append(val_dte_error)
    axes[0][1].scatter(f'depth={i}', val_dte_error, label = f'{val_dte_error}', marker = 'o', linestyle = 'None')
    axes[0][1].annotate(f'{round(val_dte_error, 7)}', (f'depth={i}',val_dte_error), textcoords='offset points', xytext = (-5, 5), ha= 'center')
    axes[1][1].scatter(f'depth={i}', tr_dte_error, label = f'{val_dte_error}', marker = 'o', linestyle = 'None')
    axes[1][1].annotate(f'{round(tr_dte_error, 7)}', (f'depth={i}',tr_dte_error), textcoords='offset points', xytext = (-5, 5), ha= 'center')
    
        
# this block plots the real win rates and errors and tidies the plots up
axes[0][0].scatter(['Hanhwa Life eSports', 'Nongshim RedForce', 'OK BRION'], validation_wrs, label = 'True win rates', marker = 'o', linestyle = 'None') 
axes[1][0].scatter(tr_teams, win_rates, label = 'True win rates', marker = 'o', linestyle = 'None') 
axes[0][0].legend() 
axes[1][0].legend()
axes[0][0].set_yticks(yticks100)
axes[1][0].set_yticks(yticks100)
axes[0][1].set_yticks(yticks_error)
axes[1][1].set_yticks(yticks_error)
axes[0][0].set_title('Decision tree regression validation set predicted win rates with depths 1-10')
axes[1][0].set_title('Decision tree regression training set predicted win rates with depths 1-10')
axes[0][1].set_title('Decision tree regression validation mean squared error with maxdepth 1-10')
axes[1][1].set_title('Decision tree regression training mean squared error with maxdepth 1-10')
dte = DecisionTreeClassifier(max_depth=10)    # initialize DTC class, fit the model
dte.fit(training_statistics, win_rates)
dte_test_pred = dte.predict(pred_stats)
test_dte_error = mean_squared_error(dte_test_pred, test_wrs)

plt.figure(figsize=(18, 6))    # visualize results on a plot
plt.plot(test_teams, dte_test_pred, marker = 'o', linestyle = 'None', label = 'LCK Spring 2024 predicted win rates')
plt.plot(test_teams, test_wrs, marker = 'o', linestyle = 'None', label = 'LCK Spring 2024 true win rates')
plt.yticks(yticks100)
plt.title(f'LCK Spring 2024 predicted win rates vs. true win rates\nDecision tree regression, maximum depth = 10\nMSE:{test_dte_error}')
plt.legend()
