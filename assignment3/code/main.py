import numpy as np
import pandas as pd
import math
import time
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfTransformer

#exploring the dataset
df = pd.read_csv("English_Dataset.csv")

#shuffle the dataset
df = df.sample(frac=1)

df.head(10)

#all the categories and their counts on dataset
print(df["Category"].value_counts())

#split data into train and test samples
X = df["Text"].to_numpy()
y = df["Category"].to_numpy()

#%70 training, %30 testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

#label_values
label_values = df["Category"].unique()
#ngram parameters
ngrams = { "unigram":(1,1), "bigram": (2,2)}


# get occurences words list of each category
def most_occurence_values(df, labels, unmeaningful_words=set()):
    occurence_dict = {}
    # for each category
    for l in labels:
        data = df[df["Category"] == l]
        word_counts = {}
        total_count = 0
        # for each text in category
        for text in data["Text"].to_numpy():
            word_list = text.split()
            # for each word in text
            for word in word_list:
                # increase the count of word
                if word in unmeaningful_words:
                    continue
                elif word in word_counts:
                    word_counts[word] += 1
                    total_count += 1
                else:
                    word_counts[word] = 1
                    total_count += 1
        # sort words list according to their occurence count
        word_counts = dict(sorted(word_counts.items(), key=lambda item: item[1], reverse=True))
        # set occurence counts to occurence frequency
        for word in word_counts:
            word_counts[word] = word_counts[word] / total_count

        occurence_dict[l] = word_counts
    return occurence_dict


# print first n words that has highest occurence frequency
def get_n_occurence_values(df, labels, n=3, unmeaningful_words=set()):
    # occurence dictionary
    occurences = most_occurence_values(df, label_values, unmeaningful_words)
    # for each category
    for l in occurences:
        # words and their frequencies as array
        word_list = list(occurences[l].keys())
        word_list_freqs = list(occurences[l].values())
        print("For {} label , most occurence {} words are :".format(l, n))
        for i in range(n):
            print("{:<12}  {:>12}".format(word_list[i], ("frequency : %" + str(word_list_freqs[i] * 100))))
        print()

get_n_occurence_values(df,label_values, 3)

#words that has no meanings at all
my_set = ("-", "s", "t", "m", "o", "mr","said", "just","new")       # adding these words into stop words
unmeaningful_words = set(ENGLISH_STOP_WORDS).union(my_set)

#occurences without unnecessary words
get_n_occurence_values(df,label_values, 3,unmeaningful_words)


# method for obtaining train and text word matrixes and also vocabulary list
def get_CountVectorizer_results(x_train, x_test, ngram_range, stopwords=frozenset()):
    # initilazing vectorizer
    vect = CountVectorizer(ngram_range=ngram_range, stop_words=stopwords)

    # train and text vectors
    train_cv = vect.fit_transform(x_train)
    vocabulary_list = vect.get_feature_names_out()
    test_cv = vect.transform(x_test)

    # train and test matrixes
    train_words_matrix = train_cv.toarray()
    test_words_matrix = test_cv.toarray()

    # return values
    return train_words_matrix, test_words_matrix, vocabulary_list


# fitting train data for later predicting phase
def naive_bayes_fit(x_train, y_train, train_words_matrix, vocabulary_list, labels):
    wordCountsOfLabels = {}  # each word counts in each category
    total_counts_of_labels = {}  # total numbers of each category
    # each categorys probabilities as log value
    logs_of_each_labels = {}  # basically P(category)

    for l in labels:
        # defaultdict used for hadling unseen word in test matrix
        wordCountsOfLabels[l] = defaultdict(lambda: 0)
        total_counts_of_labels[l] = len(x_train[np.where(y_train == l)])
        logs_of_each_labels[l] = math.log(total_counts_of_labels[l] / len(y_train))  # P(category)=category/total

    # set word counts in each category for wordCountsOfLabels
    # for each sample in training
    for i in range(len(y_train)):
        l = y_train[i]  # category of text
        # words that appeared in train sample
        avaliable_words = np.array(np.where(train_words_matrix[i] != 0))[0]
        for j in avaliable_words:
            wordCountsOfLabels[l][vocabulary_list[j]] += train_words_matrix[i][j]

    # return values
    return wordCountsOfLabels, total_counts_of_labels, logs_of_each_labels


def naive_bayes_predict(test_words_matrix, wordCountsOfLabels, total_counts_of_labels, logs_of_each_labels,
                        labels, vocabulary_list):
    preds = []  # predictions
    # for each test sample
    for i in range(test_words_matrix.shape[0]):
        # each categories probabilty values
        label_scores = {}
        # initialize each categories probabilty values with P(category)
        for l in logs_of_each_labels:
            label_scores[l] = logs_of_each_labels[l]

        # words in test sample
        words_arr = []
        for j in np.where(test_words_matrix[i] != 0)[0]:
            words_arr.append(vocabulary_list[j])

        # for each word in test sample
        for word in words_arr:
            # for each category
            for l in labels:
                # laplace smoothing used for 0 probability , P(text|category)
                # dominator = count of word in this category
                # denominator=total count of category
                dominator = wordCountsOfLabels[l][word] + 1  # 1 is laplace smoothing
                denominator = total_counts_of_labels[l] + len(
                    vocabulary_list)  # len(vocabulary_list) is laplace smoothing
                # add to the label scores, we are adding because we use log probability
                label_scores[l] += math.log(dominator / denominator)
        # get category that has probability value closes to 0
        preds.append(max(label_scores, key=label_scores.get))
    return preds

#training and predicting-unigram- Stopwords not  used- TD-IDF not used
st = time.time()
train_words_matrix, test_words_matrix, vocabulary_list = get_CountVectorizer_results(X_train, X_test, ngrams["unigram"])

wordCountsOfLabels, total_counts_of_labels, logs_of_each_labels = naive_bayes_fit(X_train, y_train,
                                                                                train_words_matrix, vocabulary_list,
                                                                               label_values)

y_preds_unigram_part2 = naive_bayes_predict(test_words_matrix, wordCountsOfLabels, total_counts_of_labels, logs_of_each_labels,
                        label_values , vocabulary_list)

print("Training and Prediction Time : {} sn".format((time.time()-st)))

#training and predicting-bigram- Stopwords not  used- TD-IDF not used
st = time.time()
train_words_matrix, test_words_matrix, vocabulary_list = get_CountVectorizer_results(X_train, X_test, ngrams["bigram"])

wordCountsOfLabels, total_counts_of_labels, logs_of_each_labels = naive_bayes_fit(X_train, y_train,
                                                                                train_words_matrix, vocabulary_list,
                                                                                label_values)

y_preds_bigram_part2=naive_bayes_predict(test_words_matrix, wordCountsOfLabels, total_counts_of_labels, logs_of_each_labels,
                        label_values , vocabulary_list)

print("Training and Prediction Time : {} sn".format((time.time()-st)))

def get_N_TfidfTransformer_values(df,category,n=10,ngram_range=(1,1), stopwords=frozenset()):
    #get the category dataset
    data = df[df["Category"]==category]
    data = data["Text"]
    #count vectorizer
    vect = CountVectorizer(ngram_range=ngram_range, stop_words=stopwords)
    data_cv = vect.fit_transform(data)
    vocabulary = vect.get_feature_names_out()
    #TfidfTransformer
    tdif_Transformer = TfidfTransformer()
    #passsing countvectorizer data array to TfidfTransformer
    tdif_Transformer.fit_transform(data_cv.toarray())
    #sort idf_ array
    sorted_tdif = np.argsort(tdif_Transformer.idf_ )
    #return last 10 elements of idf_ as presence and first 10 elements as absence
    return [vocabulary[i] for i in sorted_tdif][-n:], [vocabulary[i] for i in sorted_tdif][:n]


# List the 10 words whose presence and absence most strongly predicts that the article
# belongs to specific category for each five categories -unigram
n = 10
presencesUnigram = []
absencesUnigram = []
presencesBigram = []
absencesBigram = []
for l in label_values:
    # presences and absences of category for unigram
    presenceUn, absenceUn = get_N_TfidfTransformer_values(df, l, n, ngrams["unigram"])
    # presences and absences of category for bigram
    presenceBi, absenceBi = get_N_TfidfTransformer_values(df, l, n, ngrams["bigram"])
    presencesUnigram.append(presenceUn)
    absencesUnigram.append(absenceUn)
    presencesBigram.append(presenceBi)
    absencesBigram.append(absenceBi)

pd.DataFrame(presencesUnigram, columns=[i for i in range(1,11)], index=label_values)
pd.DataFrame(presencesBigram, columns=[i for i in range(1,11)], index=label_values)
pd.DataFrame(presencesBigram, columns=[i for i in range(1,11)], index=label_values)
pd.DataFrame(absencesBigram, columns=[i for i in range(1,11)], index=label_values)


def get_CountVectorizer_results_using_TfidfTransformer(x_train, x_test, ngram_range, stopwords=frozenset()):
    # count vectorizer results
    vect = CountVectorizer(ngram_range=ngram_range, stop_words=stopwords)
    train_cv = vect.fit_transform(x_train)
    test_cv = vect.transform(x_test)

    tfidf_Transformer = TfidfTransformer()
    # train and test matrixes using TfidfTransformer with CountVectorizer result matrixes
    train_fidf_cv = tfidf_Transformer.fit_transform(train_cv.toarray())
    vocabulary_list = tfidf_Transformer.get_feature_names_out()
    test_fidf_cv = tfidf_Transformer.transform(test_cv.toarray())

    train_words_matrix = train_fidf_cv.toarray()
    test_words_matrix = test_fidf_cv.toarray()

    # return train matrix ,test matrix and vocabulary list
    return train_words_matrix, test_words_matrix, vocabulary_list

st = time.time()

train_words_matrix, test_words_matrix, vocabulary_list = get_CountVectorizer_results_using_TfidfTransformer(X_train, X_test,
                                                                                       ngrams["unigram"])

wordCountsOfLabels, total_counts_of_labels, logs_of_each_labels = naive_bayes_fit(X_train, y_train,
                                                                                train_words_matrix, vocabulary_list,
                                                                                label_values)

y_preds_unigram_part3_a = naive_bayes_predict(test_words_matrix, wordCountsOfLabels, total_counts_of_labels, logs_of_each_labels,
                        label_values , vocabulary_list)

print("Training and Prediction Time : {} sn".format((time.time()-st)))

st = time.time()
train_words_matrix, test_words_matrix, vocabulary_list = get_CountVectorizer_results_using_TfidfTransformer(X_train, X_test,
                                                                                       ngrams["bigram"])

wordCountsOfLabels, total_counts_of_labels, logs_of_each_labels = naive_bayes_fit(X_train, y_train,
                                                                                train_words_matrix, vocabulary_list,
                                                                                label_values)

y_preds_bigram_part3_a = naive_bayes_predict(test_words_matrix, wordCountsOfLabels, total_counts_of_labels, logs_of_each_labels,
                        label_values , vocabulary_list)

print("Training and Prediction Time : {} sn".format((time.time()-st)))

#List the 10 words whose presence and absence most strongly predicts that the article
#belongs to specific category for each five categories. Stopwords are removed.
n = 10
presencesUnigramStopwords=[]
absencesUnigramStopwords=[]
presencesBigramStopwords=[]
absencesBigramStopwords=[]
for l in label_values:
    #presences and absences of category for unigram
    presenceUn, absenceUn = get_N_TfidfTransformer_values(df, l, n, ngrams["unigram"], stopwords=ENGLISH_STOP_WORDS)
    #presences and absences of category for bigram
    presenceBi, absenceBi = get_N_TfidfTransformer_values(df, l, n, ngrams["bigram"], stopwords=ENGLISH_STOP_WORDS)
    presencesUnigramStopwords.append(presenceUn)
    absencesUnigramStopwords.append(absenceUn)
    presencesBigramStopwords.append(presenceBi)
    absencesBigramStopwords.append(absenceBi)

pd.DataFrame(presencesUnigramStopwords, columns=[i for i in range(1,11)], index=label_values)
pd.DataFrame(presencesBigramStopwords, columns=[i for i in range(1,11)], index=label_values)

pd.DataFrame(absencesUnigramStopwords, columns=[i for i in range(1,11)], index=label_values)
pd.DataFrame(absencesBigramStopwords, columns=[i for i in range(1,11)], index=label_values)

st = time.time()
train_words_matrix, test_words_matrix, vocabulary_list = get_CountVectorizer_results(X_train, X_test,
                                                                                       ngrams["unigram"],
                                                                                  stopwords=ENGLISH_STOP_WORDS)

wordCountsOfLabels, total_counts_of_labels, logs_of_each_labels = naive_bayes_fit(X_train, y_train,
                                                                                train_words_matrix, vocabulary_list,
                                                                                label_values)

y_preds_unigram_part3_b_stopwords = naive_bayes_predict(test_words_matrix, wordCountsOfLabels, total_counts_of_labels, logs_of_each_labels,
                        label_values , vocabulary_list)

print("Training and Prediction Time : {} sn".format((time.time()-st)))

st = time.time()
train_words_matrix, test_words_matrix, vocabulary_list = get_CountVectorizer_results(X_train, X_test,
                                                                                       ngrams["bigram"],
                                                                                  stopwords=ENGLISH_STOP_WORDS)

wordCountsOfLabels, total_counts_of_labels, logs_of_each_labels = naive_bayes_fit(X_train, y_train,
                                                                                train_words_matrix, vocabulary_list,
                                                                                label_values)

y_preds_bigram_part3_b_stopwords = naive_bayes_predict(test_words_matrix, wordCountsOfLabels, total_counts_of_labels, logs_of_each_labels,
                        label_values , vocabulary_list)

print("Training and Prediction Time : {} sn".format((time.time()-st)))

print("Part 2 Results : ")
print("Unigram Accuracy - TF-IDF is not used : ", accuracy_score(y_test, y_preds_unigram_part2))
print("Bigram Accuracy - TF-IDF is not used : ", accuracy_score(y_test, y_preds_bigram_part2))

print("Part 3-a Results : ")
print("Unigram Accuracy - TF-IDF used :", accuracy_score(y_test,y_preds_unigram_part3_a))
print("Bigram Accuracy - TF-IDF used : ", accuracy_score(y_test,y_preds_bigram_part3_a))

print("Part 3-b Results : ")
print("Unigram Accuracy - Stopwords are removed: ", accuracy_score(y_test,y_preds_unigram_part3_b_stopwords))
print("Bigram Accuracy - Stopwords are removed : ", accuracy_score(y_test,y_preds_bigram_part3_b_stopwords))

