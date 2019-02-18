/** RESOURCES
* https://scrubadub.readthedocs.io/en/stable/
* http://www.nltk.org/api/nltk.stem.html
* http://www.nltk.org/howto/stem.html
*https://www.nltk.org/book/ch02.html
*https://stackoverflow.com/questions/18193253/what-exactly-is-an-n-gram
*http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html
*http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
*http://scikit-learn.org/stable/modules/svm.html
*http://scikit-learn.org/stable/modules/naive_bayes.html
*http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
*http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
*http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
*/


import pandas as pd
import csv

#open csv file 
result = pd.read_csv("/Users/caoshengnan/Desktop/social_remove_retweet.csv")

#remove names
#pip install scrubadub
import scrubadub
def remove_name_lower(column):
	link_removed_string = re.sub(r'( https://t\.co/)[a-zA-Z0-9]{10}.*', '', column)
    nameless_string = scrubadub.clean(link_removed_string)
    remove_scrubbed_identifiers = re.sub(r'{{.*}}', '', nameless_string)
    lower_string = remove_scrubbed_identifiers.lower()
    return lower_string

#write no_name list to a new column
result['remove_names_lower_content'] = result.apply(lambda x:remove_name_lower(x.Contents), axis=1) 

#lemmatize
import re
#remove punctuations and repair abbreviation
def lemma(dataset, field):
     dataset[field]= dataset[field].str.replace(r"(it|he|she|that|this|there|here)(\'s)", " is")
     dataset[field]= dataset[field].str.replace(r"(?<=[a-zA-Z])\'ve", " have")
     dataset[field]= dataset[field].str.replace(r"(?<=[a-zA-Z])\'s", "")
     dataset[field]= dataset[field].str.replace(r"(?<=s)\'s?", "")
     dataset[field]= dataset[field].str.replace(r"can't", "can not")     	 dataset[field]= dataset[field].str.replace(r"won't", "will not")
     dataset[field]= dataset[field].str.replace(r"(?<=[a-zA-Z])n\'t", " not")
     dataset[field]= dataset[field].str.replace(r"(?<=[a-zA-Z])\'d", " would")
     dataset[field]= dataset[field].str.replace(r"(?<=[a-zA-Z])\'ll", " will")
     dataset[field]= dataset[field].str.replace(r"(?<=[I|i])\'m", " am")
     dataset[field]= dataset[field].str.replace(r"(?<=[a-zA-Z])\'re", " are")
     dataset[field]= dataset[field].str.replace(r"[^A-Za-z0-9]", " ")
     dataset[field]= dataset[field].str.replace(r"[?\!\"]", " ")
     return dataset

result = lemma(result, "remove_names_lower_content")

#remove stopwords
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
stopwords = open("/Users/caoshengnan/Desktop/stopwords.txt",'r').read().split()
st_removed_list = []
for i in result.remove_names_lower_content:
     raw = str(i)
     tokens = tokenizer.tokenize(raw)
     stopped_tokens = [p for p in tokens if not p in stopwords]
     st_removed_list.append(stopped_tokens)
 
result['remove_st_content'] = st_removed_list


#frequency distribution
from collections import Counter
text_list = []
for row, index in result.iterrows():
        
        text_string = index[‘remove_st_content']
        
        text_list.append(text_string)       
flat_skip_list = [item for sublist in text_list for item in sublist]   
ctr = Counter(flat_skip_list)
#get frequency distribution, add some words to stop words list

#stemm text
stemmed_list = []
for i in result.remove_st_content:
     stemmed_line = []
     for word in i:
         stemmed_line.append(stemmer.stem(word))
     stemmed_list.append(stemmed_line)
 
result['stemmed_content'] = stemmed_list

result.to_csv("/Users/caoshengnan/Desktop/preproccessed_data.csv")

# convert list to string
string_list = []
for i in result.remove_st_content:
     string_list.append(' '.join(i))
 
result['string_content'] = string_list

##Bag of Words / feature extract/ Split into training & test set
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
#column to list
list_corpus = result[‘string_content’].tolist()
list_labels = result[‘Sentiment’].tolist()

#split function
X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels,test_size=0.2,random_state=40)

#feature extraction
count_vectorizer = CountVectorizer()
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)


#fit classifiers

#fit LogisticRegression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1.0, class_weight='balanced', solver='newton-cg',multi_class='multinomial', n_jobs=-1, random_state=40)
clf.fit(X_train_counts, y_train)
y_predicted_counts = clf.predict(X_test_counts)

#Evaluation/Scores
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
def get_metrics(y_test, y_predicted):  
     precision = precision_score(y_test, y_predicted, pos_label=None,average='weighted')
     recall = recall_score(y_test, y_predicted, pos_label=None,average='weighted')
     f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
     accuracy = accuracy_score(y_test, y_predicted)
     return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

Outputs：accuracy = 0.827, precision = 0.848, recall = 0.827, f1 = 0.836


#fit Naive bayes— MultinomialNB classifier

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)
y_predicted_counts = clf.predict(X_test_counts)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
def get_metrics(y_test, y_predicted):
     precision =precision_score(y_test,y_predicted,pos_label=None,average='weighted')
     recall = recall_score(y_test, y_predicted, pos_label=None,average='weighted')
     f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
     accuracy = accuracy_score(y_test, y_predicted)
     return accuracy, precision, recall, f1
 
accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

Outputs： accuracy = 0.852, precision = 0.819, recall = 0.852, f1 = 0.819

## LinearSVC

from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = LinearSVC(random_state=0)
clf.fit(X_train_counts, y_train)

LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
     verbose=0)

y_predicted_counts = clf.predict(X_test_counts)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
def get_metrics(y_test, y_predicted):
    precision = precision_score(y_test, y_predicted, pos_label=None,average='weighted')
    recall = recall_score(y_test, y_predicted, pos_label=None,average='weighted')
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1
 
accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
Outputs： accuracy = 0.860, precision = 0.844, recall = 0.860, f1 = 0.850

# fit SGDClassifier
 from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(X_train_counts, y_train)
y_predicted_counts = clf.predict(X_test_counts)
print(classification_report(y_test, y_predicted_counts))


#save classifier
import pickle
save_nb = pickle.dumps(clf)

#load classfier
clf_nb = pickle.load(save_nb)
list = clf_nb.predict(X_train_counts)

#MLP Classifier
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100,100,100,100), random_state=1)
clf.fit(X_train_counts, y_train)
y_predicted_cv = clf.predict(X_test_counts)
print(classification_report(y_test, y_predicted_counts))


#Grid Search

#MLP Classifier with grid search to optimize parameters
import numpy as np
from sklearn.model_selection import GridSearchCV
parameters = {'solver': ['lbfgs'], 'warm_start': [True], 
              'alpha': [1e-5, 1e-4, 1e-3], 'hidden_layer_sizes':[1,5,1000], 
              'random_state':[0,1,2]}

clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1, scoring='f1_weighted')
clf.fit(X_train_cv, y_train)

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
parameters = {'solver': ['lbfgs'],'alpha': [1e-5, 1e-4],'hidden_layer_sizes':[1,5,100],'random_state':[0,1,2]}
clf = GridSearchCV(MLPClassifier(), parameters)
clf.fit(X_train_counts, y_train)
GridSearchCV(cv=None, error_score='raise',
       estimator=MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(100,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False),
       fit_params=None, iid=True, n_jobs=1,
       param_grid={'solver': ['lbfgs'], 'alpha': [1e-05, 0.0001], 'hidden_layer_sizes': [1, 5, 100], 'random_state': [0, 1, 2]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring=None, verbose=0)

print(clf.best_score_)
0.8579814429085401
print(clf.best_params_)
{'alpha': 1e-05, 'hidden_layer_sizes': 100, 'random_state': 0, 'solver': 'lbfgs'}

# grid search logistic regression

from sklearn.model_selection import GridSearchCV
parameters = {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
clf = LogisticRegression(penalty='l2', random_state=777, max_iter=10000, tol=10)
gs = GridSearchCV(clf, parameters)
gs.fit(X_train_counts, y_train)

print(gs.best_score_)
0.8445370195038818
print(gs.best_params_)
{'C': 0.001}

# grid search linear SVC
from sklearn.svm import LinearSVC

parameters = {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 'max_iter':[10, 100, 1000, 10000]}
clf = LinearSVC(penalty='l2', random_state=777,tol=10)
gs = GridSearchCV(clf, parameters)
gs.fit(X_train_counts, y_train)
GridSearchCV(cv=None, error_score='raise',
       estimator=LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=777, tol=10, verbose=0),
       fit_params=None, iid=True, n_jobs=1,
       param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [10, 100, 1000, 10000]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring=None, verbose=0)
print(gs.best_score_)
0.8646089755728081
print(gs.best_params_)
{'C': 0.1, 'max_iter': 10}

parameters = {‘C’:[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'max_iter':[1, 10, 100, 1000], ‘intercept_scaling’: [ 1, 10, 100], ‘random_state’: [1, 10, 100], ‘tol’: [ 1, 10, 100, 1000]}

gs = GridSearchCV(LinearSVC(), parameters)
gs.fit(X_train_counts, y_train)


parameters = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'max_iter': [1,10,100,1000], 'intercept_scaling':[1,10,100], 'random_state': [1, 10, 100], 'tol': [1,10,100]}
gs = GridSearchCV(LinearSVC(), parameters)
gs.fit(X_train_counts, y_train)
GridSearchCV(cv=None, error_score='raise',
       estimator=LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0),
       fit_params=None, iid=True, n_jobs=1,
       param_grid={'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'max_iter': [1, 10, 100, 1000], 'intercept_scaling': [1, 10, 100], 'random_state': [1, 10, 100], 'tol': [1, 10, 100]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring=None, verbose=0)
print(gs.best_score_)
0.8712365082370763
print(gs.best_params_)
{'C': 0.1, 'intercept_scaling': 100, 'max_iter': 1000, 'random_state': 1, 'tol': 1}


# grid search SVC

>>> from sklearn.svm import SVC
>>> parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'kernel':['linear'], 'kernel':['rbf'], 'gamma':[0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
>>> gs = GridSearchCV(SVC(), parameters)
>>> gs.fit(X_train_counts, y_train)
GridSearchCV(cv=None, error_score='raise',
       estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False),
       fit_params=None, iid=True, n_jobs=1,
       param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring=None, verbose=0)
>>> print(gs.best_score_)
0.8699110017042226
>>> print(gs.best_params_)
{'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}

# grid search SDG
>>> parameters = {'alpha': [0.0001, 0.000001], 'penalty':['l2', 'elasticnet'], ‘max_iter':[10,50,80,100]}
>>> gs = GridSearchCV(SGDClassifier(), parameters)
>>> gs.fit(X_train_counts, y_train)
GridSearchCV(cv=None, error_score='raise',
       estimator=SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', max_iter=None, n_iter=None,
       n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
       shuffle=True, tol=None, verbose=0, warm_start=False),
       fit_params=None, iid=True, n_jobs=1,
       param_grid={'alpha': [0.0001, 1e-06], 'penalty': ['l2', 'elasticnet'], ‘max_iter': [10, 50, 80, 100]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring=None, verbose=0)
>>> print(gs.best_score_)
0.8655557659534179
>>> print(gs.best_params_)
{'alpha': 0.0001, 'n_iter': 10, 'penalty': 'elasticnet'}



#Vader Analysis

#pip install twython 
#pip install vaderSentiment
import string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
result = pd.read_csv("/Users/caoshengnan/Desktop/social_remove_retweet.csv")
messages = result['Contents'].tolist()
sid = SentimentIntensityAnalyzer()
for x in messages: 
    ss = sid.polarity_scores(x)
    if ss["compound"] == 0.0: 
        summary["neutral"] +=1
    elif ss["compound"] > 0.0:
        summary["positive"] +=1
    else:
        summary["negative"] +=1

print(summary)
{'positive': 2095, 'neutral': 3886, 'negative': 621}
# create 3 empty lists
 pos_list = []
 neu_list = []
 neg_list = []
#write text to corresponding list
 for x in messages:
     ss = sid.polarity_scores(x)
     if ss["compound"] == 0.0:
             neu_list.append(x)
     elif ss["compound"] > 0.0:
             pos_list.append(x)
     else:
             neg_list.append(x)

#write lists to csv columns
from pandas.core.frame import DataFrame
df_pos = pd.DataFrame({'Sentiment':pos_list})
df_pos.to_csv("/Users/caoshengnan/Desktop/sentiment.csv ")
df_neu = pd.DataFrame({'Neutral':neu_list})
df_neu.to_csv("/Users/caoshengnan/Desktop/sentiment.csv", mode = 'a')
df_neg = pd.DataFrame({'Negative':neg_list})
df_neg.to_csv("/Users/caoshengnan/Desktop/sentiment.csv", mode = 'a')

