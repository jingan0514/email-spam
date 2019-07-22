# Name: Gan Jin
# UNI: gj2297

import os
import re
import random
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt

ps = PorterStemmer()
vectorizer = CountVectorizer()
stop_words = set(stopwords.words("english"))

# load file
def load(directory):
    file_path = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            path = f"{directory}/{file}"
            file_path.append(path)
    
    return file_path

# split the dataset into training data and testing data
def seperate_train_test(file_list, ratio=0.7):
    train = random.sample(file_list, round(len(file_list)*ratio))
    test = [f for f in file_list if f not in train]

    return train, test

def clean_email(f):
    lines = f.readlines()
    regex_list = [r".*-.-.-.*", r"Subject:.*", r"subject :.*", r"from :.*", r"to :.*", r"cc :.*"]            
    for r in regex_list:
        regex = re.compile(r)
        lines = [l for l in lines if not regex.match(l)]
    
    tokenized_word = []
    for l in lines:
        tokenized_word += word_tokenize(l.strip().lower())
                    
    filtered_word = []
    for w in tokenized_word:
        if w not in stop_words:
            filtered_word.append(w)

    regex = re.compile(r"\w+[^\d+]")
    filtered_word = [w for w in filtered_word if regex.match(w)]
    
    # stem
    stemmed_words = []
    for w in filtered_word:
        stemmed_words.append(ps.stem(w))

    return stemmed_words

def cv(file_path):
    words = []
    for file in file_path:
        try:
            with open(file) as f:
                stemmed_words = clean_email(f)
                words += stemmed_words
        except:
                pass

    dictionary = Counter(words)
    dictionary = dictionary.most_common(3000)
    
    common_word = []
    for key, value in dictionary:
        common_word.append(key)
    
    cv = CountVectorizer(vocabulary=common_word)
    
    return cv

def Euclidean_represent(file_path, cv):
    content_list = []
    for file in file_path:
        try:
            with open(file) as f:
                stemmed_words = clean_email(f)
                content = ""

                for w in stemmed_words:
                    content += f"{w} "

            content_list.append(content)
        
        except:
            pass

    vector = cv.fit_transform(content_list).toarray()
    return vector


# directory of ham and spam emails
ham_directory = "enron1/ham"
spam_directory = "enron1/spam"

# load two kinds of emails
ham_email = load(ham_directory)
spam_email = load(spam_directory)

# split data into train and test data
ham_train, ham_test = seperate_train_test(ham_email)
spam_train, spam_test = seperate_train_test(spam_email)

# compute countvector 
ham_cv = cv(ham_train)
spam_cv = cv(spam_train)


# compute vector
ham_vector = Euclidean_represent(ham_train, ham_cv)
spam_vector = Euclidean_represent(spam_train, spam_cv)
ham_test_v = Euclidean_represent(ham_test, ham_cv)
spam_test_v = Euclidean_represent(spam_test, spam_cv)


# ham test
result_ham_1 = []
result_ham_2 = []
result_ham_inf = []
for vector in ham_test_v:
    
    # distance to ham vector     
    distance1_ham = []
    distance2_ham = []
    distance_infinite_ham = []
    
    for v in ham_vector:
        
        d1 = 0
        d2 = 0
        d_infinite = []
        
        for i in range(len(v)):
            d1 += abs(vector[i] - v[i])
            d2 += pow(vector[i] - v[i], 2)
            
            d_infinite.append(abs(vector[i] - v[i]))
        
        d_inf = max(d_infinite)
            
        distance1_ham.append(d1)
        distance2_ham.append(d2)
        distance_infinite_ham.append(d_inf)
        
    # distance to spam vector
    distance1_spam = []
    distance2_spam = []
    distance_infinite_spam = []
    
    for v in spam_vector:
        
        d1 = 0
        d2 = 0
        d_infinite = []
        
        for i in range(len(v)):
            d1 += abs(vector[i] - v[i])
            d2 += pow(vector[i] - v[i], 2)
            
            d_infinite.append(abs(vector[i] - v[i]))
        
        d_inf = max(d_infinite)
            
        distance1_spam.append(d1)
        distance2_spam.append(d2)
        distance_infinite_spam.append(d_inf)

    if min(distance1_ham) <= min(distance1_spam):
        result_ham_1.append("Ham")
    else:
        result_ham_1.append("Spam")


    if min(distance2_ham) <= min(distance2_spam):
        result_ham_2.append("Ham")
    else:
        result_ham_2.append("Spam")


    if min(distance_infinite_ham) <= min(distance_infinite_spam):
        result_ham_inf.append("Ham")
    else:
        result_ham_inf.append("Spam")


# spam test
result_spam_1 = []
result_spam_2 = []
result_spam_inf = []
for vector in spam_test_v:
    
    # distance to ham vector     
    distance1_ham = []
    distance2_ham = []
    distance_infinite_ham = []
    
    for v in ham_vector:
        
        d1 = 0
        d2 = 0
        d_infinite = []
        
        for i in range(len(v)):
            d1 += abs(vector[i] - v[i])
            d2 += pow(vector[i] - v[i], 2)
            
            d_infinite.append(abs(vector[i] - v[i]))
        
        d_inf = max(d_infinite)
            
        distance1_ham.append(d1)
        distance2_ham.append(d2)
        distance_infinite_ham.append(d_inf)
        
    # distance to spam vector
    distance1_spam = []
    distance2_spam = []
    distance_infinite_spam = []
    
    for v in spam_vector:
        
        d1 = 0
        d2 = 0
        d_infinite = []
        
        for i in range(len(v)):
            d1 += abs(vector[i] - v[i])
            d2 += pow(vector[i] - v[i], 2)
            
            d_infinite.append(abs(vector[i] - v[i]))
        
        d_inf = max(d_infinite)
            
        distance1_spam.append(d1)
        distance2_spam.append(d2)
        distance_infinite_spam.append(d_inf)

    if min(distance1_ham) <= min(distance1_spam):
        result_spam_1.append("Ham")
    else:
        result_spam_1.append("Spam")


    if min(distance2_ham) <= min(distance2_spam):
        result_spam_2.append("Ham")
    else:
        result_spam_2.append("Spam")


    if min(distance_infinite_ham) <= min(distance_infinite_spam):
        result_spam_inf.append("Ham")
    else:
        result_spam_inf.append("Spam")



# L1
ham_ham_1 = result_ham_1.count("Ham")
ham_spam_1 = result_ham_1.count("Spam")
spam_ham_1 = result_spam_1.count("Ham")
spam_spam_1 = result_spam_1.count("Spam")


# L2
ham_ham_2 = result_ham_2.count("Ham")
ham_spam_2 = result_ham_2.count("Spam")
spam_ham_2 = result_spam_2.count("Ham")
spam_spam_2 = result_spam_2.count("Spam")


# L_inf
ham_ham_inf = result_ham_inf.count("Ham")
ham_spam_inf = result_ham_inf.count("Spam")
spam_ham_inf = result_spam_inf.count("Ham")
spam_spam_inf = result_spam_inf.count("Spam")