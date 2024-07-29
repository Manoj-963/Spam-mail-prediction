
					SPAM MAIL PREDICTION
Mails are of two types: spam and non-spam also known as ham mails.
1.	Mail data
2.	Data pre-processing
3.	Train test split
4.	New mail is given to the trained model and then we predict if it is spam or ham
For training we use logistic regression model because for binary classification (spam or ham) it is best preferred.

Step 1:
Importing the dependencies (importing the required libraries)-
1.	import numpy as np
for creating numpy arrays

2.	import pandas as pd
for using dataframes which are useful in making the data in structured manner
the csv files are made into a more structured table content with the help of data frames

3.	 from sklearn.model_selection import train_test_split
the train_test_split function is useful for training and testing of data 

4.	 from sklearn.feature_extraction.text import Tfidfvectorizer
the tfidfvectorizer is useful in generation of feature vectors (numerical values)
this is done because the numerical data is easily understood by the machine than the text information 

5.	 from sklearn.linear_model import LogisticRegression
the training data is useful for training the logistic regression model 
the logistic regression is best for binary classification

6.	 from sklearn.metrics import accuracy_score
the test data is useful for prediction of the accuracy score
this accuracy score is useful for evaluating the model
Step 2:
Data collection and pre-processing
1.	rawdata = pd.read_csv(‘ path ‘)
loading the data from csv file to pandas dataframe
2.	print(rawdata)
for printing the raw data 

3.	now we have to replace the null values (missing values) with a null string
maildata = rawdata.where((pd.notnull(rawdata)), ‘ ‘)
the where function is useful for a condition and here it helps in filling the missing values with and empty string (‘ ‘)

4.	printing the first five rows
print(maildata.head())

printing the no. of rows and columns
print(maildata.shape)

o/p: (x,y) where x is the no. of rows and y is the no. of columns

We have labels for all mails i.e. ham or spam

Now we do LABEL ENCODING
WE ENCODE THE LABEL TO NUMERICAL VALUES
“REPLACE TEXT VALUE WITH NUMERICAL VALUE”
REPLACE HAM WITH 1
REPLACE SPAM WITH 0
Step 3: 
1.	Label encoding
maildata.loc[maildata[‘category’] == ‘spam’,’category’,] = 0
maildata.loc[maildata[‘category’] == ‘ham’,’category’,] = 1
2.	Separating the data into to texts and labels
x = mailData[‘Message’]
y = mailData[‘Category’]

3.	Separating the data into training data and test data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 3)
The x_train,x_test,y_train,y_test are arrays which hold the mails as given by the train_test_split function
Here the test_size tells us that the 20% of the data is used for testing purpose
The random_state is optional 
The train_test_split function works in different manner i.e., for each time of its execution it separates different mails into training and testing data 
In order to make sure that it separates the same mails in all cases we can use the random_state
The random_state can hold any numerical value.
Step 4:
Feature Extraction
1.	Transform the text data to feature vectors that can be used as input to the logistic regression model
feature_extraction = TfidfVectorizer(min_df = 1,stop_words = ‘english’, lowercase=True)
Here we load the vectorizer which is used for generation of feature vectors into the feature_extraction variable. The min_df tells us that the least occurring words can be ignored. The stop_words=’english’ makes sure that words such as is, did, are, as etc can be ignored as they can commonly exist in all mails so for them, we don’t need to create feature vectors. The lowercase converts all the words into lowercase for better understanding of data.
x_train_features = feature_extraction.fit_transform(x_train)
A new variable x_train_features would be useful for storing the x_train values (textual form) as numerical values i.e., as feature vectors. Here we use two functions fit and transform. Fit for fitting the vectorizer and transform for converting the text data to feature vectors.
x_test_features = feature_extraction.transform(x_test)
2.	Convert the y_train and y_test values to integer type (by default they are of object type)
y_train = y_train.astype(‘int’)
y_test = y_test.astype(‘int’)

Step 5:
Training the model: Logistic Regression
model = LogisticRegression()
Train the logistic regression model with the training data
model.fit(x_train_features,y_train)

Step 6:
Evaluating the trained model:
1.	Prediction on training data-
prediction_on_training_data = model.predict(x_train_features)
accuracy_on_training_data = accuracy_score(y_train,prediction_on_training_data)
print(“Accuracy on trained data:”, accuracy_on_training_data)

2.	Prediction on test data-
prediction_on_test_data = model.predict(x_test_features)
accuracy_on_test_data = accuracy_score(y_test,prediction_on_test_data)
print(“Accuracy on test data:”, accuracy_on_test_data)

Step 7: 
A new mail is given as input and validated if it is spam or ham 
Example:
	input_mail = [“mail_data”]
	input_mail_features = feature_extraction.transform(input_mail)
	prediction = model.predict(input_mail_features)
	print(prediction)
The prediction can have 2 values,0 or 1, and is a list.
	if prediction==1:
		print(“Ham mail”)
	else:
		print(“Spam mail”)






THE END






