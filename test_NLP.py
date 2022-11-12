# import required packages
import numpy as np
from tensorflow import keras
from train_NLP import *
from sklearn.metrics import accuracy_score


if __name__ == "__main__": 
	# 1. Load your saved model
	model = keras.models.load_model('./models/NLP_Model.h5')
	
	# 2. Load your testing data
	data = IMDB_Data()
	train , test = data.get_data("./data/aclImdb")
	X_train, X_test, Y_train, Y_test = data.process_data(train, test)

	# 3. Run prediction on the test data and print the test accuracy
	y_pred = model.predict(X_test)
	y_pred = np.array(list(map(lambda x:1 if (x>0.5) else 0, y_pred)))
	test_acc = accuracy_score(y_pred, Y_test.astype('float32'))
	print(f"Test accuracy: {test_acc}")