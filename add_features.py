import csv
import pandas as pd
import numpy as np
import sys
import argparse
from nltk.tokenize import sent_tokenize, word_tokenize,RegexpTokenizer
from nltk.corpus import words as english_words, stopwords
import re
import nltk
from nltk.stem import PorterStemmer
import datetime
import time


def main(file_name):
	if file_name is None:
		raise Exception("Enter filename here")
		return

	'''
	Loading the data file
	'''
	data_file = "data/%s" %(file_name)
	df = pd.read_csv(data_file,error_bad_lines = False)

	'''
	Fetching the total number of emoticons in the tweet
	'''		
	#loading the emoticons from the emoticons list found in /data/positive.txt and /data/negative.txt
	positive_list = get_positive_emoticons()
	negative_list = get_negative_emoticons()

	#counting number of positive and negative emoticons and storing them as additional column
	num_positive_emoticons = count_emoticon_occurences(df['tweet'],positive_list)
	df['num_positive_emoticons'] = pd.Series(num_positive_emoticons)
	num_negative_emoticons = count_emoticon_occurences(df['tweet'],negative_list)
	df['num_negative_emoticons'] = pd.Series(num_negative_emoticons)


	'''
	Fetching total no of exclamations
	'''

	num_exclamations = count_characters(df['tweet'], '!')
	df['num_exclamations'] = pd.Series(num_exclamations)


	'''
	Fetching total no of hastags
	'''
	num_hashtags = count_characters(df['tweet'], '#')
	df['num_hashtags'] = pd.Series(num_hashtags)


	'''
	Fetching total no of question marks
	'''
	num_question_marks = count_characters(df['tweet'],'?')
	df['num_question_marks'] = pd.Series(num_question_marks)

	'''
	Fetching total no of hyperlinks
	'''
	num_links = count_characters(df['tweet'],'http')
	df['num_links'] = pd.Series(num_links)

	'''
	Cleaning the tweets
	It removes stopwords, hyperlinks, special characters and single digit characters
	Next it stems the tweet using NLTK PorterStemmer technique
	'''
	df['cleaned_tweet'] = df['tweet'].apply(clean_my_tweets)


	'''
	Converting the tweets into a feature set using the bag-of-words model
	For that we are considering the top 3000 words occuring in the tweet and making a feature set for them
	Stores the output in /data/bow_with_features.csv
	'''
	wordlist = make_wordlist(df)
	out_file = "data/bow_with_features_%s" %(file_name)
	df = bag_of_words(wordlist,df,out_file)
	df.to_csv('data/bow_with_features.csv')





def make_wordlist(df):
	print ("Converting to worlist")
	all_words = []
	for tweet in df['cleaned_tweet']:
		for w in tweet.split():
			all_words.append(w)
	all_words = nltk.FreqDist(all_words)

	word_features = list(all_words.keys())[:3000]

	min_occurrences = 3
	max_occurences = 2000
	word_df = pd.DataFrame(data={"word": [k for k, v in all_words.most_common() if min_occurrences < v < max_occurences],
                                     "occurrences": [v for k, v in all_words.most_common() if min_occurrences < v < max_occurences]},
                               columns=["word", "occurrences"])

	wordlist = [k for k, v in all_words.most_common() if min_occurrences < v < max_occurences]
	return wordlist





def bag_of_words(wordlist,df,out_file):
	print ("Converting to bag-of-words")

	label_column = ["label"]																																																									
	new_cols = [col for col in df.columns if col.startswith("num_")]
	columns = label_column + new_cols + list(map(lambda w: w + "_bow",wordlist))

	rows = []
	for idx in df.index:
		current_row = []
		current_label = df.loc[idx, "Sentiment"]
		current_row.append(current_label)


		for _,col in enumerate(new_cols):
			current_row.append(df.loc[idx,col])

		tokens = set((df.loc[idx, 'cleaned_tweet']).split())
		for _, word in enumerate(wordlist):
			current_row.append(1 if word in tokens else 0)
		rows.append(current_row)

	data_model = pd.DataFrame(rows, columns=columns)
	return data_model

def clean_my_tweets(inp):
	inp = inp.lower()
	usernames = re.findall(r'@[\w\.:]+',inp)
	out = ' '.join(filter(lambda x: x.lower() not in usernames, inp.split()))

	return clean_stopwords(out)

def clean_stopwords(inp):
	stop_words = set(w.lower() for w in stopwords.words())
	out = ' '.join(filter(lambda x: x.lower() not in stop_words, inp.split()))
	return remove_hyperlinks(out)

def remove_hyperlinks(inp):
	out = ' '.join(word for word in inp.split() if not word.startswith(('www.','http')))
	return remove_special_chars(out)

def remove_special_chars(inp):
	out = re.sub('[^a-zA-Z \n]', '', inp)
	return (stem_tweet(out))

def stem_tweet(inp, stemmer=nltk.PorterStemmer()):
	out = ' ' .join (map(lambda str: stemmer.stem(str), inp.split()))
	return remove_single_letters(out)

def remove_single_letters(inp):
	out = ' '.join( [w for w in inp.split() if len(w)>1] )
	return (out)


def get_positive_emoticons():
	positive_list =[]

	content = open('data/positive.txt')
	for line in content:
		positive_list.append(line.rstrip())
	return (positive_list)

def get_negative_emoticons():
	negative_list =[]

	content = open('data/negative.txt')
	for line in content:
		negative_list.append(line.rstrip())
	return (negative_list)

def count_emoticon_occurences(inp, search_list):
	
	pos_series = []
	for _,row in enumerate(inp):
		agg = 0
		for _,i in enumerate(search_list):
			emo_count = row.count(i)
			agg = agg + emo_count

		pos_series.append(agg)
	return (pos_series)

def count_characters(inp,character):
	out_series = []
	for _,row in enumerate(inp):
		agg = row.count(character)
		out_series.append(agg)

	return out_series


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Convert data into a feature set')
	parser.add_argument('-f', '--file_name', help='Enter path of training data', required=True)
	args = parser.parse_args()
	main(args.file_name)


