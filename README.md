Twitter Sentiment Analyzer

This repo contains the code that will classify tweets as either 'positive' or 'negative' using various machine learning models such as Naive baeyes, Random Forest and others. Rather than relying on older algorithms such as VADER and Textblob, this method can model a classifier from scratch which also takes into account the presence of features such as emoticons, punctuations, exclamations, hashtags and other characters to determine the sentiment of the tweet.

Usage:

Getting the data:

A training set of data from Stanford was used to train the model. Another training set can be used to train the model as well. The path for the training data is data/train.csv and the format is as follows.

Similarily the testing data (tweets which are to be predicted) as stored in the path /data/test.csv.
Any other data set can be used and placed in the above path to obtain the prediciton.
The format for the test data is as follows:


Extracting features:

The following features were added to the existing dataset.
a. No. of positive emoticons(/data/positive.txt)
b. No. of negative emoticons(/data/negative.txt)
c. No. of exclamations
d. No. of hashtags
e. No. of question marks
f. No. of hyperlinks

Prior to fitting the model and using machine learning algorithms for training, we need to represent it in a bag of words model. Plus to add the above features to the training and the testing data set, the following script is run on the script.

python add_features.py

Prediction:

After the Analytical Base Table was ready for sentiment classification, various machine-learning algorithms can be used to classify the tweets as positive or negative.
Run the script to check the results of the prediction.

python predict.py

