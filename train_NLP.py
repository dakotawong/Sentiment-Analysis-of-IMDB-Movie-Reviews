# import required packages
import glob
import numpy as np
import os
import pandas as pd
from pkg_resources import add_activation_listener
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow import keras
import json
from sklearn.model_selection import train_test_split


# Class to format and process the dataset
class IMDB_Data():
	def __init__(self) -> None:
		pass

	# Reads data from either test or train set
	def read_data(self, dir):
		# Lists to store features and labels
		X = []
		y = []

		# Both test and train sets have subfolders 'pos' and 'neg'
		subfolders = ["pos", "neg"]
		for subfolder in subfolders:
			path = os.path.join(dir, subfolder, "*")
			review = ([1] if subfolder == "pos" else [0])

			# Iteratively read files and append contents to our lists
			for file in glob.glob(path, recursive = False):
				file_content = open(file, "r", encoding = "utf8").read()
				X.append(file_content)
				y.append(review)
		
		return X, y
	
	# Return a dataframe for both train and test data
	def get_data(self, dir):
		input_train, label_train = self.read_data(dir + "/train")
		input_test, label_test = self.read_data(dir + "/test")

		header_input = ['review']
		header_label = ['sentiment']
		
		df_input_train = pd.DataFrame(input_train, columns=header_input)
		df_input_test = pd.DataFrame(input_test, columns=header_input)
		df_label_train = pd.DataFrame(label_train, columns=header_label)
		df_label_test = pd.DataFrame(label_test, columns=header_label)
		
		df_train = pd.concat((df_input_train, df_label_train), axis=1)
		df_test = pd.concat((df_input_test, df_label_test), axis=1)
		
		return df_train, df_test
	
	# Dataframe (df) has column target (df[target]) made all lowercase
	def to_lowercase(self, df, target):
		df[target] = df[target].apply(lambda x: " ".join(x.lower() for x in x.split()))
		return df
	
	# Dataframe (df) has column target (df[target]) have all puncuation removed
	def remove_punct(self, df, target):
		df[target] = df[target].str.replace('[^\w\s]' , '')
		return df
	
	# Dataframe (df) has column target (df[target]) have all stopwords removed
	def remove_stopwords(self, df, target):
		stop = stopwords.words('english')
		df[target] = df[target].apply(lambda x: " ".join([x for x in x.split() if x not in (stop)]))
		return df
	
	# Dataframe (df) has column target (df[target]) is lemmatized
	def lematization(self, df, target):
		lemmatizer = WordNetLemmatizer()
		df[target] = df[target].apply(lambda x: " ".join(lemmatizer.lemmatize(x) for x in x.split()))
		return df
	
	# Tokenizes target column of dataframe
	def tokenizer(self, df_train, df_test, target, output=None):
		if output is None: 
			output = target 
		converter = text.Tokenizer(num_words=10000)
		converter.fit_on_texts(df_train[target])
		converter.fit_on_texts(df_train[target])

		df_train[output] = converter.texts_to_sequences(df_train[target])
		df_test[output] = converter.texts_to_sequences(df_test[target])

		return df_train, df_test
	
	# Addeds padding to target and returns the padded target and labels as numpy arrays
	def padding(self, df, target_pad, label, seq_len=None):
		reviews = df[target_pad].to_numpy()
		labels = df[label].to_numpy()

		if seq_len == None:
			len_seq = [len(seq) for seq in reviews]
			avg_seq = int(sum(len_seq) / len(len_seq))
			seq_len = avg_seq

		reviews_padded = sequence.pad_sequences(reviews, padding='post', maxlen=seq_len, truncating='post')		

		return reviews_padded, labels, seq_len
	
	# Pre-process a dataframe returning results as numpy array
	def process_data(self, df_train, df_test):
		df_train = self.to_lowercase(df_train, "review")
		df_train = self.remove_punct(df_train, "review")
		df_train = self.remove_stopwords(df_train, "review")
		df_train = self.lematization(df_train, "review")

		df_test = self.to_lowercase(df_test, "review")
		df_test = self.remove_punct(df_test, "review")
		df_test = self.remove_stopwords(df_test, "review")
		df_test = self.lematization(df_test, "review")
		
		df_train, df_test = self.tokenizer(df_train, df_test, "review")
		
		X_train, Y_train, seq_len = self.padding(df_train, "review", "sentiment")
		X_test, Y_test, seq_len = self.padding(df_test, "review", "sentiment", seq_len=seq_len)
		
		X_train, Y_train = shuffle(X_train, Y_train)
		X_test, Y_test = shuffle(X_test, Y_test)

		return X_train, X_test, Y_train, Y_test


if __name__ == "__main__":

	# 1. load your training data

	data = IMDB_Data()
	train , test = data.get_data("./data/aclImdb")
	X_train, X_test, Y_train, Y_test = data.process_data(train, test)

	# Split the train set into training and test sets (for training purposes)
	X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)

	# 2. Train your network
	# 		Make sure to print your training loss and accuracy within training to show progress
	# 		Make sure you print the final training accuracy

	input_dim = len(X_train[0])

	model = Sequential([
		layers.Embedding(10000, 16, input_length=input_dim),
		layers.LSTM(128),
		layers.Dense(20, activation='relu'),
		layers.Dense(1,activation='sigmoid')
	])

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	history = model.fit(X_train, Y_train.astype('float32'), epochs=10, validation_data=(X_test, Y_test.astype('float32')))

	# 3. Save your model
	model.save('./models/NLP_Model.h5')