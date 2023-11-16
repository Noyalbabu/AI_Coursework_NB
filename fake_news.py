import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import string 

# import the fake and real news datasets
data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')

data_fake.head()
data_true.head()

# Add a target class column to indicate whether the news is real or fake
data_fake["class"] = 0 
data_true["class"] = 1

#Remove last 10 rows for manual testing 
data_fake_manual_testing = data_fake.tail(10)
for i in range(23480,23470,-1):
    data_fake.drop([i], axis = 0, inplace = True)

data_true_manual_testing = data_true.tail(10)
for i in range(21416,21406,-1):
    data_true.drop([i], axis = 0, inplace = True)

data_fake_manual_testing['class'] = 0
data_true_manual_testing['class'] = 1

#Output the manual testing data
data_fake_manual_testing.head(10)
data_true_manual_testing.head(10)   

#Merging the manual testing data in single dataset 
data_merge = pd.concat([data_fake , data_true], axis = 0) 

#Merge the true and fake news data into a single dataframe
data = data_merge.drop(["title", "subject","date"], axis = 1)
data.isnull().sum()

#Randomly shuffle the dataframes so that the model does not learn the order of the data
data = data.sample(frac = 1)
data.reset_index(inplace = True)
data.drop(["index"], axis = 1, inplace = True)

#Function to preprocess the text
#Below shown special characters are removed from the text
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text) 
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)

    return text

data['text'] = data['text'].apply(wordopt)

#Defining dependent and independent variables
x = data['text']        #Independent variable
y = data['class']       #Dependent variable

#Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

#Convert text into vectors using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

#Logistic Regression model
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)
#Accuracy score for Logistic Regression model
LR.score(xv_test, y_test)
#Print classification report. The classification_report function builds a text report showing the main classification metrics
print(classification_report(y_test, pred_lr))

#Decision Tree Classifier model
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)
#Accuracy score for Decision Tree model
DT.score(xv_test, y_test)
#Print classification report. 
print(classification_report(y_test, pred_dt))

#Gradient Boosting Classifier model
from sklearn.ensemble import GradientBoostingClassifier
GB = GradientBoostingClassifier(random_state=0)
GB.fit(xv_train, y_train)
pred_gb = GB.predict(xv_test)
#Accuracy score for Gradient Boosting Classifier model
GB.score(xv_test, y_test)
#Print classification report. 
print(classification_report(y_test, pred_gb))

#Random Forest Classifier model
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(random_state = 0)
RF.fit(xv_train, y_train)
pred_rf = RF.predict(xv_test)
#Accuracy score for Random Forest Classifier model
RF.score(xv_test, y_test) 
#Print classification report.
print(classification_report(y_test, pred_rf))  

def output_label(dataset):
    if dataset == 0:
        return "Fake News"
    elif dataset == 1:
        return "True News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)
    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGB Prediction: {} \nRF Prediction: {}".format(output_label(pred_LR[0]), 
                                                                                                            output_label(pred_DT[0]), 
                                                                                                            output_label(pred_GB[0]), 
                                                                                                            output_label(pred_RF[0])))
print("Enter the news text")
news = str(input())
manual_testing(news)


