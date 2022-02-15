# Author: Kelvin Lu
# HW1: KNN Classifer Project

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier 

import nltk
import re
import string 
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
import time
from statistics import mode
import matplotlib.pyplot as plt


training_path = 'train_hw1.txt'
test_path = 'test_hw1.txt' 
output_path = 'output.txt'

def get_ratings(raw_data):
    # Obtain training rankings at the start of each line. 
    # Assumptions: Assume all reviews starts with the rating.

    rating_list = np.zeros(raw_data.shape[0], dtype = 'int8')
    data = raw_data.itertuples(index=True, name=None)
    for index, row in data:
        if '-' in row[:1]:
            rating_list[index] = -1
        else:
            rating_list[index] = 1
    return rating_list

def preprocess_helper(text):
    """
        Text - The string that is being preprocessed.
        Removes all digits, punctuations, and tabs from the text.
        Lowercase the text.
        Return text.
    """

    # Lower case Text
    # Remove HTML Text
    # Remove Websites
    # Remove Numbers
    # Remove tabs
    # Replace punctuation with white space
    # Remove extra spaces. 
    
    text = re.sub(r'<[^<>]*>', '', text).lower()
    text = re.sub(r'\S+\.com[^ ]*', '', text)
    text = re.sub(r'\S+\.net[^ ]*', '', text)
    text = text.translate(text.maketrans('', '', string.digits))
    text = re.sub('\t', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(' +', ' ', text).strip()
  
    return text

def stem_helper(text):
    """
        Returns the stem version of the sentence. 
    """
    tokenized_words = word_tokenize(text)
    stop_words = stopwords.words('english')
    
    # stemmer = SnowballStemmer(language='english')
    # stemmer = PorterStemmer() 
    # stemmer = LancasterStemmer()
    lemmatizer = WordNetLemmatizer()
    # print(snow_stemmer.stem('managed'))
    

    # tokenized_words = [stemmer.stem(word) for word in tokenized_words]
    tokenized_words = [lemmatizer.lemmatize(word) for word in tokenized_words]
    # if word not in stop_words
    return tokenized_words
    # return text

class knn_classifer():  
    def __init__(self, k):
        """
            Classify reviews in raw data using k nearest neighb
            k - Number of neighbors used to classify data point
        """
        self.k = k
        self.training = None
        self.labels = None

    def fit(self, x_train, y_train):
        """
            x-train - tf-IDF vectors of training data
            y-train - Labels of training data
        """
        self.training = x_train
        # Rating
        self.labels = y_train

    def predict(self, x_test):
        output = np.zeros(x_test.shape[0], dtype = 'int8')
        # print(type(x_test))
        # print(x_test.shape)
        # print(self.training.shape)
        # print(x_test[0])
        # print(self.training[0])

        print('Distance calculating...')
        # method = 1
        # distances = cosine_similarity(self.training, x_test)
        method = 0 
        distances = euclidean_distances(x_test, self.training)

        print(distances.shape)
        # print(type(distances))
        # print('Demension of dist-matrix', distances.shape)
        return self.generate_predictions(distances, flag = method)
        # for row in distances[0]:
            # print(row)
        # return output

    def generate_predictions(self, dist_matrix, flag):
        # Return ordered list (closest to furthest) of k neighbors

        print('Genreating predictions')
        output = np.zeros(dist_matrix.shape[0], dtype = 'int8')

        # print(dist_matrix)
        # print('--------')
        # print(type(self.labels))
        for row in range(len(dist_matrix)):
            # A list that contains the indices of the sorted distance of the array. 
            sorted_row = np.argsort(dist_matrix[row])

            # Cosine-Similarity - Reverse the row 
            if(flag == 1):
                print('Cosine List Reversed')
                sorted_row = np.flip(sorted_row)

            # Find the first k neighbor from the sorted_row. 
            neighbors = list()
            for i in range(self.k):
                neighbors.append(self.labels[sorted_row[i]])

            # print(neighbors)
            # Generate a prediction. 
            output[row] = mode(neighbors)

        # print(output)
        return output
        

def write_helper(num):
    if num == 1:
        return '+1'
    else:
        return '-1'

def run_K(train_data, training_vectors, test_tfidf):
    # k_neighbors = 10
    k_neighbors = math.floor(math.sqrt(train_data.shape[0]))
    if(k_neighbors % 2 == 0):
        k_neighbors = k_neighbors + 1 

    print('Predicting...')

    print('K = ',k_neighbors)
    knn = knn_classifer(k= k_neighbors)
    knn.fit(training_vectors.todense(), train_data['Ratings'])
    return knn.predict(test_tfidf.todense())

    # print(test_data.head())

def main():
    # Entry main function
    # Read the input training data and give it a rating.

    start = time.time()

    sample_size = 15000
    # , nrows = sample_size
    print('Reading Training and Test Data ...')
    # Read the training data and give its rating. 
    train_data = pd.read_table(training_path, sep="\n",
                             header=None, names=['Reviews'], dtype=str, nrows = sample_size)
    train_data['Ratings'] = get_ratings(train_data)

    test_data = pd.read_table(test_path, sep="\n",
                             header=None, names=['Reviews'], dtype=str, nrows = sample_size)

    print('Proprocessing ...')
    train_data['Reviews'] = train_data['Reviews'].apply(lambda x: preprocess_helper(x))
    test_data['Reviews'] = test_data['Reviews'].apply(lambda x: preprocess_helper(x))

    # print(train_data.head(50))

    #  max_df = 0.70
    # min_df=50, max_df = 0.70,
    print('Vectorizing Training Data ...')
    vectorizer = TfidfVectorizer(max_features = 10000, tokenizer = stem_helper, lowercase = False)
    training_vectors = vectorizer.fit_transform(train_data['Reviews'])
    test_tfidf = vectorizer.transform(test_data['Reviews'])

    print('Training vectors dimensions - ', training_vectors.shape)
    print('Test vectors dimensions - ', test_tfidf.shape)
    # print(vectorizer.vocabulary_)

    # k_range = range(1,k_neighbors)
    # k_range = [3,25,51,75,101,123]
    # k_scores = []
    # for k in k_range:
        # build classifier here
        # knn = KNeighborsClassifier(n_neighbors = k)
        # scores = cross_val_score(knn, training_vectors, train_data['Ratings'] ,cv=5,scoring='accuracy')
        # print ('k:',k,' mean:',scores.mean(), ' std:', scores.std())
        # k_scores.append(scores.mean()) 
    
    # plt.plot(k_range, k_scores)
    # plt.xlabel('Value of K for KNN')
    # plt.ylabel('Cross-Validated Accuracy')
    # plt.plot()


    # Run the classifer 
    test_data['Ratings'] = run_K(train_data, training_vectors, test_tfidf)

    print('Writing...')
    # Converts ratings into string for formatting purposes. 
    test_data['Output'] = test_data['Ratings'].apply(lambda x: write_helper(x))

    file = test_data['Output'].to_csv(output_path, sep = '\n', index = False, header = False)
    

    end = time.time()
    print('Run-time', end - start)

if __name__ == "__main__":
    main()


