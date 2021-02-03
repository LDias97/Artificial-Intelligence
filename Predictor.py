""" 
Notes: 
Run With: python Predictor.py <filename>
Example: python Predictor.py EOD-MSFT.csv
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class PredictionAgent:

    # Import Data from CSV
    data = pd.read_csv(sys.argv[1])
    print(data)
    num_rows = len(data.index)
    # print('Number of Rows', num_rows)
    # print(data['Adj_Close'])

    # Calculate features and percent change
    data['Open-Close'] = (data.Open - data.Close) / data.Open
    data['High-Low'] = (data.High - data.Low) / data.Low
    data['Percent_Change'] = data['Adj_Close'].pct_change()

    # Make sure all values are equally weighted using the pandas rolling method
    data['std_size'] = data['Percent_Change'].rolling(5).std()  # Standard Deviation Size
    data['ret_size'] = data['Percent_Change'].rolling(5).mean() # Return Size
    data.dropna(inplace=True) # Drops rows where values are missing 

    # X is the input variable
    x = data[['Open-Close', 'High-Low', 'std_size', 'ret_size']]

    # Y is the target or output variable
    # Output = 1 if next day closing price is > than todays, else Output = -1
    # 1 = Buy stock, -1 = Sell stock
    y = np.where(data['Adj_Close'].shift(-1) > data['Adj_Close'], 1, -1)

    # Total dataset length after rolling
    dataset_length = data.shape[0] # Size of File - 5
    print('Dataset Length: ', dataset_length)

    # Training dataset length
    split = int(dataset_length * 0.80) # dataset_length * Training percentage
    print('Training Split Value: ', split)

    # Train and Test the dataset
    # xtrain, xtest = x[:split], x[split:]
    # ytrain, ytest = y[:split], y[split:]
    # print(xtrain.shape, xtest.shape)
    # print(ytrain.shape, ytest.shape)
    
    # print(xtrain)
    # print(xtest)
    # print(ytrain)
    # print(ytest)


    # Train and Test the dataset using sklearn values
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.80, test_size = 0.2) # Test = 20%
    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)
    # print('x_train: ', x_train)
    # print('x_test: ', x_test)
    # print('y_train: ', y_train)
    # print('y_test: ', y_test)
    # print(x_train)
    # print(x_test)
    # print(y_train)
    # print(y_test)


    # Random Forest Regression Algorithm using sklearn
    # -----------------------------------------------------------------------------------------------------
    # from sklearn.ensemble import RandomForestRegressor

    # rf_regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
    # rf_regressor.fit(x_train, y_train)
    # rf_y_pred = rf_regressor.predict(x_test)

    # Visualising the Random Forest Regression results 
    # Scatter plot for original data 
    # plt.scatter(x, y, color = 'blue')   
    # plt.plot(x, rf_y_pred, color = 'green')  
    # plt.title('Random Forest Regression') 
    # plt.xlabel('Values') 
    # plt.ylabel('Predictions') 
    # plt.show()

    # prediction = regressor.predict(x_test)
    # errors = abs(prediction - y_test)
    

    # Random Forest Classifier Algorithm using sklearn
    # -----------------------------------------------------------------------------------------------------
    from sklearn.ensemble import RandomForestClassifier
    rf_classifier = RandomForestClassifier(random_state = 5)
    train_model = rf_classifier.fit(x_train, y_train)

    from sklearn.metrics import accuracy_score
    print('Correct Prediction (%): ', accuracy_score(y_test, train_model.predict(x_test), normalize = True) * 100.0)

    # Classification Report
    from sklearn.metrics import classification_report
    report = classification_report(y_test, train_model.predict(x_test))
    print("Classification Report: ")
    print(report)

    # Daily return diagram
    data['strategy_returns'] = data.Percent_Change.shift(-1) * train_model.predict(x)
    data.strategy_returns[split:].hist()
    plt.title('Random Forest Classifier - Histogram Report')
    plt.xlabel('Strategy returns (%)')
    plt.show()


    # Naive Bayes Classifier Algorithm using sklearn
    # -----------------------------------------------------------------------------------------------------
    from sklearn.naive_bayes import GaussianNB
    nb_model = GaussianNB()
    nb_model.fit(x_train, y_train)
    y_pred = nb_model.predict(x_test)
    print("Naive Bayes Results:", y_pred)

    # show_plot = y_pred.plot.
    # Evaluate model
    nb_accuracy = accuracy_score(y_test, y_pred) * 100
    print("Naive Bayes Accuracy: ", nb_accuracy)


# The code below is ours but using help from the website below to understand necessary parameters
# https://builtin.com/data-science/random-forest-algorithm
# -----------------------------------------------------------------------------------------------------

    # Visualising the Random Forest Regression results 
    # Scatter plot for original data 
    # plt.scatter(x, y, color = 'blue')   
    
    # # plot predicted data 
    # plt.plot(X_grid, regressor.predict(X_grid), color = 'green')  
    # plt.title('Random Forest Regression') 
    # plt.xlabel('Values') 
    # plt.ylabel('Predictions') 
    # plt.show()



# The code below is from a GeeksForGeeks Tutorial and is being used to learn and test for now
# https://www.geeksforgeeks.org/random-forest-regression-in-python/
# -----------------------------------------------------------------------------------------------------

# Prints the 'Open' Column
#x = data.iloc[:, 1:2].values
#print(x)

# Prints the 'High' Column Horizontally
#y = data.iloc[:, 2].values
#print(y)

# SKLearn Random Forest Regression to the dataset
# Import regressor
# from sklearn.ensemble import RandomForestRegressor

# regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
# regressor.fit(x, y)

# Y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))  # test the output by changing values 

# Visualising the Random Forest Regression results 
  
# arange for creating a range of values 
# from min value of x to max  
# value of x with a difference of 0.01  
# between two consecutive values 
# X_grid = np.arange(min(x), max(x), 0.01)  
  
# reshape for reshaping the data into a len(X_grid)*1 array,  
# i.e. to make a column out of the X_grid value                   
# X_grid = X_grid.reshape((len(X_grid), 1)) 
