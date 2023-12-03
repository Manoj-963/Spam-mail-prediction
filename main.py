import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#data collection and pre-processing
raw_mail_data = pd.read_csv('/content/mail_data.csv')

#filling the missing values with a null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')

# mail_data.head() for displaying the first five rows
# mail_data.shape for displaying the size of the dataset

#label encoding
mail_data.loc[mail_data['Category']=='ham','Category',] = 1
mail_data.loc[mail_data['Category']=='spam','Category',] = 0

#splitting the data into texts and labels
x = mail_data['Message']
y = mail_data['Category']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 4)

#feature_extraction
feature_extraction = TfidfVectorizer(min_df = 1,stop_words = 'english',lowercase = True)

x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)

#converting ytrain and ytest to integer values
y_train = y_train.astype('int')
y_test = y_test.astype('int')

#Training the logistic regression model
model = LogisticRegression()
model.fit(x_train_features,y_train)

#evaluating the training data
prediction_on_training_data = model.predict(x_train_features)
accuracy_on_training_data = accuracy_score(y_train,prediction_on_training_data)

#evaluating the test data
prediction_on_test_data = model.predict(x_test_features)
accuracy_on_test_data = accuracy_score(y_test,prediction_on_test_data)

input_mail = ["Your mail data"]
input_mail_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_mail_features)

if prediction == 1:
  print("Ham mail")
else:
  print("Spam mail")





