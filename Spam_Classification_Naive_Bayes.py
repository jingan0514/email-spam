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

def dictionary(file_path):
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

    return dictionary

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

# # of ham and spam emails
n_ham = len(ham_email)
n_spam = len(spam_email)

# probability of ham and spam email
ham_percent = n_ham /(n_ham + n_spam)
spam_percent = n_spam /(n_ham + n_spam)

# split data into train and test data
ham_train, ham_test = seperate_train_test(ham_email)
spam_train, spam_test = seperate_train_test(spam_email)

# compute countvector 
ham_dictionary = dictionary(ham_train)
spam_dictionary = dictionary(spam_train)

# calculate the total words in ham emails
total_words = 0
for key, value in ham_dictionary:
    total_words += value

# calculate frequency of each word in ham emails
ham_word_p = {}    
for key, value in ham_dictionary:
    ham_word_p[key] = value / total_words *1000

# claculate the total words in spam emails
total_words = 0
for key, value in spam_dictionary:
    total_words += value

# claculate the frequency of each word in spam emails
spam_word_p = {}    
for key, value in spam_dictionary:
    spam_word_p[key] = value / total_words *1000

# test for ham email
result_ham = []
for file in ham_test:
    try:
        with open(file) as f:
            data = clean_email(f)
            p_w_s = 1
            p_w_h = 1
            for w in data:
                try:
                    word_in_ham =  ham_word_p[w]
                    word_in_spam =  spam_word_p[w]
                except:
                    pass

                p_w_h *= word_in_ham
                p_w_s *= word_in_spam

            p_spam = (p_w_s*spam_percent) / (p_w_s*spam_percent + p_w_h*ham_percent)
            if p_spam >= 0.5:
                result_ham.append("Spam")
            else:
                result_ham.append("Ham")
    except:
        print("open file failed")
        
ham_class = result_ham.count("Ham")
ham_error = result_ham.count("Spam")


# test for spam email
result_spam = []
for file in spam_test:
    try:
        with open(file) as f:
            data = clean_email(f)
            p_w_s = 1
            p_w_h = 1
            for w in data:
                try:
                    word_in_ham =  ham_word_p[w]
                    word_in_spam =  spam_word_p[w]
                except:
                    pass

                p_w_h *= word_in_ham
                p_w_s *= word_in_spam

            p_spam = (p_w_s*spam_percent) / (p_w_s*spam_percent + p_w_h*ham_percent)
            if p_spam >= 0.5:
                result_spam.append("Spam")
            else:
                result_spam.append("Ham")
    except:
        print("open file failed")
        
spam_class = result_spam.count("Spam")
spam_error = result_spam.count("Ham")


df = pd.DataFrame({"Category": ["Ham", "Spam"],
                   "Ham": [result_ham.count("Ham"), result_spam.count("Ham")],
                   "Spam": [result_ham.count("Spam"), result_spam.count("Spam")]})

df.set_index("Category")