from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
import random
import pickle
import pandas as pd
import numpy as np


def main():
	''' 
	Test algorithms
	This takes the feature set and tests various ML algorithms on it
	'''
	df = pd.read_csv('data/bow_with_features.csv')
	df = df.drop('Unnamed: 0', 1).copy()
	
	X = df.iloc[:, 1:].values
	Y = df.iloc[:, 0].values
	seed = 666
	random.seed(seed)
	X_train, X_test, y_train, y_test = train_test_split(X,Y,train_size=0.7, stratify=df.iloc[:, 0],random_state=seed)
	test_classifier(X_train, y_train, X_test, y_test, BernoulliNB())
	test_classifier(X_train, y_train, X_test, y_test, RandomForestClassifier(random_state=seed,n_estimators=10,n_jobs=-1))
	test_classifier(X_train, y_train, X_test, y_test,KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2))



def test_classifier(X_train, y_train, X_test, y_test, classifier):

	# Feature Scaling
	from sklearn.preprocessing import StandardScaler
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)

	print ("")
	print ("===============================================")
	classifier_name = str(type(classifier).__name__)
	print ("Testing " + classifier_name)
	# now = time()
	list_of_labels = sorted(list(set(y_train)))
	model = classifier.fit(X_train, y_train)
	# storing the model 
	filename = 'finalized_model.sav'
	pickle.dump(model, open(filename, 'wb'))


	predictions = model.predict(X_test)
	from sklearn.metrics import confusion_matrix
	cm = confusion_matrix(y_test, predictions)
	print ("==========Confusion Matrix=====================")
	print (cm)
	precision = precision_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
	recall = recall_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
	accuracy = accuracy_score(y_test, predictions)
	f1 = f1_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
	print ("=================== Results ===================")
	print ("            Negative     Positive")
	print ("F1       " + str(f1))
	print ("Precision" + str(precision))
	print ("Recall   " + str(recall))
	print ("Accuracy " + str(accuracy))
	print ("===============================================")



if __name__ == '__main__':
	main()