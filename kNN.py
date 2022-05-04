from statistics import mean, variance
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

def readfile(filename):
    maindata = []
    with open(filename, newline='') as file:
        data = csv.reader(file)
        for d in data:
            maindata.append(d)
        x_data = [t[1:] for t in maindata]
        y_data = [t[:1] for t in maindata]
    return x_data, y_data

def normalizedata(data):
    ndata = data.copy()
    for i in range(len(data)):
        for j in range(len(data[i])):
            b = max(list(zip(*data))[j]) - min(list(zip(*data))[j])
            if data[i][j] != 0 and b != 0:
                print(b)
                ndata[i][j] = (data[i][j] - min(list(zip(*data))[j]))/b
    return ndata

def kNN(x_train, x_test, y_train, k):
    predictions = []
    for te in x_test:
        edist = []
        neighbors = []
        for tr, ytr in zip(x_train, y_train):
            l = distance.euclidean(te, tr)
            edist.append((l, ytr))
        edist.sort()
        
        for i in range(k):
            neighbors.append(edist[i][1])
        predictions.append(max(neighbors))
    return predictions

def accuracy(predictions, classlabels):
    correct = 0.0
    numsamples = float(len(classlabels))
    for i in range(len(classlabels)):
        if predictions[i] == classlabels[i]:
            correct += 1
    return (correct/numsamples) * 100.0

def dataencoding(data, colnames, name):
    ohe = OneHotEncoder()
    df = pd.DataFrame(data)
    df.columns = colnames
    transformer = make_column_transformer((OneHotEncoder(), name), remainder='passthrough')
    transformed = transformer.fit_transform(df)
    transformed_df = pd.DataFrame(transformed, columns=transformer.get_feature_names())
    return transformed_df.values.tolist()

def plot(train, test, y_train, y_test):
    acclist = []
    karr = []
    std = []
    for k in range(1, 52, 2):
        predictions = []
        acc = []
        for i in range(20):  
            print("a")
            predictions = kNN(train, test, y_train, k)
            acc.append(accuracy(predictions, y_test))
        
        acclist.append(mean(acc))
        std.append(math.sqrt(variance(acc)))
        karr.append(k)

    plt.errorbar(np.array(karr),np.array(acclist),yerr=np.array(std),marker='D')
    plt.xlabel("k values")
    plt.ylabel("Accuracy")
    plt.show()


def dataset1():
    digits = datasets.load_digits(return_X_y=True)
    digits_dataset_X = digits[0]
    digits_dataset_Y = digits[1]
    x_train, x_test, y_train, y_test = train_test_split(digits_dataset_X, digits_dataset_Y, test_size=0.2)
    scaler = preprocessing.MinMaxScaler()
    # x_train = normalizedata(x_train)
    # x_test = normalizedata(x_test)
    x_train = scaler.fit_transform(x_train)
    x_test =  scaler.fit_transform(x_test)
    # plot(x_train, x_test, y_train, y_test)
    predictions = kNN(x_train, x_train, y_train, 5)
    print(accuracy(predictions, y_train)) #97.5

def dataset2():
    x, y = readfile('titanic.csv')
    colnames = ['Pclass','Sex','Age','Siblings/Spouses Aboard','Parents/Children Aboard','Fare']
    x = dataencoding(x, colnames, ['Sex'])
    x_train, x_test, y_train, y_test = train_test_split(x[1:], y[1:], test_size=0.2)
    scaler = preprocessing.MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test =  scaler.fit_transform(x_test)
    # x_train = normalizedata(x_train)
    # x_test = normalizedata(x_test)
    plot(x_train, x_test, y_train, y_test)
    # predictions = kNN(x_train, x_test, y_train, 5)
    # print(accuracy(predictions, y_test)) #67

def dataset3():
    x, y = readfile('loan.csv')
    colnames = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']
    x = dataencoding(x, colnames, ['Gender','Married','Dependents','Education','Self_Employed','Property_Area'])
    y = dataencoding(y, ['Loan_Status'], ['Loan_Status'])
    x_train, x_test, y_train, y_test = train_test_split(x[1:], y[1:], test_size=0.2)
    scaler = preprocessing.MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test =  scaler.fit_transform(x_test)
    # x_train = normalizedata(x_train)
    # x_test = normalizedata(x_test)
    predictions = kNN(x_train, x_test, y_train, 5)
    print(accuracy(predictions, y_test)) #52

def dataset4():
    x, y = readfile('parkinsons.csv')
    x_train, x_test, y_train, y_test = train_test_split(x[1:], y[1:], test_size=0.2)
    scaler = preprocessing.MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test =  scaler.fit_transform(x_test)
    # x_train = normalizedata(x_train)
    # x_test = normalizedata(x_test)
    predictions = kNN(x_train, x_test, y_train, 5)
    print(accuracy(predictions, y_test)) #85


dataset2()

# digits = datasets.load_digits(return_X_y=False)
# data, x, y = readfile('parkinsons.csv')
# print(type(stratifiedkfold(pd.DataFrame(data), 10)))