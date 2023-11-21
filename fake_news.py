import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import model_selection, svm
import re
import string 

#Function to preprocess the text
#Below shown special characters are removed from the text
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  # Use r to denote a raw string
    text = re.sub(r"\\W", " ", text)     # Double backslash to escape it properly
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

#Function to return the output label
def output_label(dataset):
    if dataset == "FAKE":
        return "Fake News"
    elif dataset == "REAL":
        return "True News"
    
#Function to manually test the news
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_SVM = SVM.predict(new_xv_test)
    return print("\n\nLR Prediction: {} \nSVM Prediction: {}".format(output_label(pred_LR[0]), 
                                                                    output_label(pred_SVM[0])))

# load the data
news_data = pd.read_csv('NEWS/news.csv')

#droping the columns which are not required
data = news_data.drop(['Unnamed: 0','title'], axis=1)
#print(data.isnull().sum())

#Randomly shuffle the 'data' so that the model does not learn the order of the data
data = data.sample(frac = 1)
data.reset_index(inplace = True)
data.drop(["index"], axis = 1, inplace = True)

#Preprocessing the text 
data['text'] = data['text'].apply(wordopt)

#Feature and lables
x = data['text']    #feature of the dataset    
y = data['label']   #label of the dataset   

#Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

#Convert text into vectors using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
xvect_train = vectorization.fit_transform(x_train)
xvect_test = vectorization.transform(x_test)

#Logistic Regression model
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(xvect_train, y_train) #xvect_train is the feature matrix of the training set and y_train is the corresponding vector of labels
pred_lr = LR.predict(xvect_test)
#Accuracy score for Logistic Regression model
print("Accuracy of Logistic Regression model:", LR.score(xvect_test, y_test)*100)
#Print classification report. The classification_report function builds a text report showing the main classification metrics
print("Classification Report of Logistic Regression:")
print(classification_report(y_test, pred_lr))

#Using Support Vector Machine to build the model
print("Performing Support Vector Machine Classification..........")
SVM = svm.SVC( C= 1.0, kernel = 'linear')
SVM.fit(xvect_train, y_train)
pred_SVM = SVM.predict(xvect_test)
#Accuracy of the model
print("Accuracy of SVM model: ", accuracy_score(y_test, pred_SVM)*100)
#Print classification report. The classification_report function builds a text report showing the main classification metrics
print("Classification Report of SVM:")
print(classification_report(y_test, pred_SVM))

#Recieving the news from user  
print("Enter the news text")
news = str(input())
manual_testing(news)