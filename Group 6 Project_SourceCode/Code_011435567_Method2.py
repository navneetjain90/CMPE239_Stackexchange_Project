import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from datetime import datetime
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.pipeline import Pipeline
from nltk.tokenize import TreebankWordTokenizer
from sklearn import linear_model
from sklearn.metrics import accuracy_score

cachedStopWords = stopwords.words("english")
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\:)"  # :)
                           u"\:P"  # :p
                           u"\:D"  # :D
                           "]+", flags=re.UNICODE)

def remove_smileys(document):
  return emoji_pattern.sub(r'', document)


def remove_digits(document):
  return re.sub(r'\d+', '', document)

def replace_url(document):
  result = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", document).split())
  result = re.sub(r"http\S+", "", result)
  return result

def replace_specialchar(document):
  result = re.sub('[^a-zA-Z\n\.]', ' ', document).replace(".", "")
  result = ' '.join(result.split())
  result = "".join(result.splitlines())
  return result.strip().lower()

def remove_stopwords(document):
  text = document.lower()
  text = ' '.join([word for word in text.split() if word not in cachedStopWords])
  temp = re.sub(r'\brt\b', '', text).strip(' ')
  temp = re.sub(r'\b\w{1,1}\b', '', temp)
  return temp

def remove_html_tags(document):
  p = re.compile(r'<.*?>')
  return re.sub(p, '', document)


def readTrain(file):
  print('reading file {}'.format(file))
  data_frame = pd.read_csv(file, names=['id', 'title', 'content', 'tags'], header=None)
  data_frame['title'] = data_frame['title'].apply(lambda x: remove_html_tags(x))
  data_frame['content'] = data_frame['content'].apply(lambda x: remove_html_tags(x))
  data_frame['tags'] = data_frame['tags'].apply(lambda x: remove_html_tags(x))
  data_frame['tags'] = data_frame['tags'].apply(lambda x: x.strip(' ').split())

  data_frame['text'] = data_frame[['title', 'content']].apply(lambda x: ''.join(x), axis=1)
  data_frame['text'] = data_frame['text'].apply(lambda x: remove_smileys(x))
  data_frame['text'] = data_frame['text'].apply(lambda x: replace_url(x))
  data_frame['text'] = data_frame['text'].apply(lambda x: remove_stopwords(x))
  data_frame['text'] = data_frame['text'].apply(lambda x: remove_digits(x))

  data_frame.drop('title', 1, inplace=True)
  data_frame.drop('content', 1, inplace=True)
  data_frame.drop('id', 1, inplace=True)
  data_frame.drop(data_frame.head(1).index, inplace=True)
  print('finished processing file : {}'.format(file))
  return data_frame

# the method to read in the data from all the csvs, concatenate them and return the X (text) & Y(tags) from the data set
def readTrainingDataSet():
  # files = ['./data/biology.csv','./data/cooking.csv','./data/crypto.csv', './data/robotics.csv']
  # files = ['./data/biology.csv']
  # files = ['./data/robotics.csv']
  # files = ['./data/cooking.csv']
  # files = ['./data/crypto.csv']
  # files = ['./data/diy.csv']
  # files = ['./data/travel.csv']
  files = ['./data/cooking_short.csv', './data/crypto_short.csv', './data/cooking_short_train.csv']
  # files = ['./data/cooking_short_train.csv']
  # files = ['./data/crypto_short.csv']
  # files = ['./data/tiny.csv']
  train_data_frames = []
  for f in files:
    train_data_frames.append(readTrain(f))
  print('Data inside Dataframe for files {}'.format(files))
  data_frame = pd.concat(train_data_frames)
  data_frame.to_csv('concat.csv')
  return data_frame['text'].values, data_frame['tags'].values, files



# method to read the test data
def readTestDataSet():
  file ='./data/test.csv'
  data_frame = pd.read_csv(file, names=['id', 'title', 'content'], nrows=1000)
  data_frame['title'] = data_frame['title'].apply(lambda x: remove_html_tags(x))
  data_frame['content'] = data_frame['content'].apply(lambda x: remove_html_tags(x))

  data_frame['text'] = data_frame[['title', 'content']].apply(lambda x: ''.join(x), axis=1)
  data_frame['text'] = data_frame['text'].apply(lambda x: remove_smileys(x))
  data_frame['text'] = data_frame['text'].apply(lambda x: replace_url(x))
  data_frame['text'] = data_frame['text'].apply(lambda x: remove_stopwords(x))
  data_frame['text'] = data_frame['text'].apply(lambda x: remove_digits(x))

  data_frame.drop('title', 1, inplace=True)
  data_frame.drop('content', 1, inplace=True)
  data_frame.drop('id', 1, inplace=True)
  data_frame.drop(data_frame.head(1).index, inplace=True)

  print('finished cleaning Test data')
  return data_frame['text'].values

#  a utility method to write the classification results to a file
import os
def write_to_file(Y_orig, Y_back, name, score, data_files, test_doc_ids):
  file_name = 'res_' + name +'_'
  file_name += str(datetime.now()).replace(' ','_')
  file_name += '.txt'
  f = open(os.path.join('/home/jayam/GIT/Transfer_Learning_Stack_Exchange/'+file_name),'w')
  f.write('Data Files : {}\n'.format(data_files))
  f.write('Score : {} \n'.format(score))
  f.write('\nNote : Doc ids refer to document nunmber in concat.csv file \n')
  f.write('Org  : Original, Prd : Predicted \n')

  f.write('\n\n')

  for i in range(len(Y_back)):
    f.write('Doc id  : {}\n'.format(test_doc_ids[i]))
    f.write('Org tag : {}\n'.format(Y_orig[i]))
    f.write('Prd tag : {}\n'.format(Y_back[i]))
  f.close()
  print('~'*60)
  print('Completed: {} , written : {}'.format(name, file_name))
  print('~'*60)

def write_to_file_test_Data_red(Y_back, name):
  file_name = 'res_' + name +'_'
  file_name += str(datetime.now()).replace(' ','_')
  file_name += '.txt'
  f = open(os.path.join('/home/jayam/GIT/Transfer_Learning_Stack_Exchange/'+file_name),'w')
  f.write('\n\n')

  for i in range(len(Y_back)):
    f.write('Prd tag : {}\n'.format(Y_back[i]))
  f.close()
  print('~'*60)
  print('Completed: {} , written : {}'.format(name, file_name))
  print('~'*60)


# a method for trying oneVsRest with SVC classifier
def oneVsRest_SVCLinear(X_train, X_test, Y_train, Y_test, word_dict, tags_dict ):
  print('Processing : oneVsRest_LogReg')
  print('-'*50)

  Y_original = Y_test
  vectorizer = CountVectorizer(min_df=1, vocabulary=word_dict)
  X_v_train = vectorizer.fit_transform(X_train)
  X_v_test = vectorizer.fit_transform(X_test)
  uniq_tags_names = list(tags_dict.keys())

  mlb = preprocessing.MultiLabelBinarizer(classes=uniq_tags_names)
  Y_train = mlb.fit_transform(Y_train)
  Y_test = mlb.fit_transform(Y_test)

  classifier = OneVsRestClassifier(SVC(kernel='linear'))
  classifier.fit(X_v_train, Y_train)
  score = classifier.score(X_v_test, Y_test)
  print('-' * 50)
  print('Score oneVsRest_SVCLinear: {}'.format(score))
  print('-' * 50)
  Y_pred = classifier.predict(X_v_test)
  Y_back = mlb.inverse_transform(Y_pred)
  write_to_file(Y_original, Y_back, 'oneVsRest_SVCLinear')

# a method for trying oneVsRest with Logistic Regression
def oneVsRest_LogReg(X_train, X_test, Y_train, Y_test, word_dict, tags_dict ):
  print('Processing : oneVsRest_LogReg')
  print('-'*50)

  Y_original = Y_test
  vectorizer = CountVectorizer(min_df=1, vocabulary=word_dict)
  X_v_train = vectorizer.fit_transform(X_train)
  X_v_test = vectorizer.fit_transform(X_test)
  uniq_tags_names = list(tags_dict.keys())

  mlb = preprocessing.MultiLabelBinarizer(classes=uniq_tags_names)
  Y_train = mlb.fit_transform(Y_train)
  Y_test = mlb.fit_transform(Y_test)

  classifier = OneVsRestClassifier(LogisticRegression(penalty='l2', C=0.01))
  classifier.fit(X_v_train, Y_train)
  score = classifier.score(X_v_test, Y_test)
  print('-' * 50)
  print('Score oneVsRest_LogReg : {}'.format(score))
  print('-' * 50)
  Y_pred = classifier.predict(X_v_test)
  Y_back = mlb.inverse_transform(Y_pred)
  write_to_file(Y_original, Y_back, 'oneVsRest_LogReg')

# a method for trying oneVsRest with Logistic Regression classifier but on TfIdf transformed data
def oneVsRest_LogReg_TfIdf(X_train, X_test, Y_train, Y_test, word_dict, tags_dict, data_files, test_doc_ids ):
  print('Processing : oneVsRest_LogReg_TfIdf')
  print('-'*50)

  Y_original = Y_test
  vectorizer = CountVectorizer(min_df=1, vocabulary=word_dict)
  X_v_train = vectorizer.fit_transform(X_train)
  X_v_test = vectorizer.fit_transform(X_test)
  transformer = TfidfTransformer(smooth_idf=False)
  X_train_tf = transformer.fit_transform(X_v_train)
  X_test_tf = transformer.fit_transform(X_v_test)

  uniq_tags_names = list(tags_dict.keys())
  mlb = preprocessing.MultiLabelBinarizer(classes=uniq_tags_names)
  Y_train = mlb.fit_transform(Y_train)
  Y_test = mlb.fit_transform(Y_test)

  classifier = OneVsRestClassifier(LogisticRegression(penalty='l2', C=0.01))
  classifier.fit(X_train_tf, Y_train)
  score = classifier.score(X_test_tf, Y_test)
  print('-' * 50)
  print('Score oneVsRest_LogReg_TfIdf : {}'.format(score))
  print('-' * 50)
  Y_pred = classifier.predict(X_v_test)
  Y_back = mlb.inverse_transform(Y_pred)
  write_to_file(Y_original, Y_back, 'oneVsRest_LogREg', score, data_files, test_doc_ids)


def oneVsRest_Pipeline(X_train, X_test, Y_train, Y_test, X_test_2P, words_dict, data_files, test_doc_ids):
  Y_original = Y_test # Keeping the Original tags aside,  will be used in comparing the predicted results with originals
  print('Processing : oneVsRest_Pipeline_BINREL')
  print('-'*50)

  # converting text to Matrix
  # vectorizer = CountVectorizer(min_df = 1, vocabulary = words_dict)
  # X_v_train = vectorizer.fit_transform(X_train)
  # X_v_test = vectorizer.fit_transform(X_test)
  #
  print('words chck : {}'.format(len(words_dict)))

  # transforming the Tags to matrix
  mlb = preprocessing.MultiLabelBinarizer()
  Y_train = mlb.fit_transform(Y_train)
  print('Y_train.shape : {}'.format(Y_train.shape))
  Y_test = mlb.fit_transform(Y_test)
  print('Y_test.shape : {}'.format(Y_test.shape))
  print('-'*50)

  # creating the SGDClassifier
  cls = linear_model.SGDClassifier(loss='hinge', alpha=1e-3,
                                   n_iter=50, random_state=None, learning_rate='optimal')

  # creating the pipeline for tranforming & predicting the data
  clf = Pipeline([
    ('vectorizer', CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize)),
    ('clf', BinaryRelevance(classifier=cls, require_dense=[False, True]))
  ])

  clf.fit(X_train, Y_train)

  # collecting the predicted results
  Y_pred = clf.predict(X_test)
  # applying the inverse transform to get the labels from the matrix result
  Y_back = mlb.inverse_transform(Y_pred)

  # calculating the score of the classifier
  score = accuracy_score(Y_test, Y_pred, normalize=True)
  print('-' * 50)
  print('Score oneVsRest_Pipeline_BINREL : {}'.format(score))
  # sending the results for writing in the files
  write_to_file(Y_original, Y_back, 'oneVsRest_Pipeline_BINREL', score, data_files, test_doc_ids)

  # now predicting to the actual test data using the trained model
  Y_pred_test_2p = clf.predict(X_test_2P)
  # getting the
  Y_pred_test_2p_back = mlb.inverse_transform(Y_pred_test_2p)
  write_to_file_test_Data_red(Y_pred_test_2p_back, 'oneVsRest_Pipeline_BINREL_Test_2P_res')

# this method splits the training data in to two parts one is 90/% of the actual data and another being the 10% of
# the data, This 10% data is used as a test data for measuring the accuracy of our trained model. Apart from this,
# it also filters out the document that contains such tags that have very low document frequency of 1. And the
#  remaining data is then divide to get the 90-10 percent split on the data.
def get_absolute_train_data(X, Y):
  words_dict = {}
  tags_dict = {}
  tags_freq = {}

  for x in X:
    for w in x.split():
      if w not in words_dict:
        words_dict[w] = len(words_dict)

  docId = 0
  for y in Y:
    for t in y:
      if t not in tags_dict:
        tags_dict[t] = len(tags_dict)
      if t not in tags_freq:
        tags_freq[t]  = []
      tags_freq[t].append(docId)
    docId += 1

  train_doc_ids = []
  for k in tags_freq:
    train_doc_ids.append(tags_freq[k][0])

  highFreq_doc_ids = set()
  low_freq_doc_ids = set()
  for k in tags_freq:
    doc_freq = len(tags_freq[k])
    if(doc_freq > 1 ):
      for t in tags_freq[k]:
        highFreq_doc_ids.add(t)
    else:
      for t in tags_freq[k]:
        low_freq_doc_ids.add(t)


  only_high_freq_doc_ids = list(highFreq_doc_ids.difference(low_freq_doc_ids))
  length = len(only_high_freq_doc_ids)

  # selecting the random 10% percent data that will be used for testing the model
  ten_percent = int(length * 0.1)
  test_doc_ids = np.random.choice(only_high_freq_doc_ids, ten_percent, replace= False)

  train_doc_ids = []
  for i in highFreq_doc_ids:
    if i not in test_doc_ids:
      train_doc_ids.append(i)

  X_abs_train = X[train_doc_ids]
  Y_abs_train = Y[train_doc_ids]

  X_abs_test = X[test_doc_ids]
  Y_abs_test = Y[test_doc_ids]

  return X_abs_train, Y_abs_train, X_abs_test, Y_abs_test, tags_dict, words_dict, test_doc_ids

def main():
  X, Y , data_files = readTrainingDataSet()
  X_test_2P = readTestDataSet()
  # Diagonise messages to be written on console
  print('-'*50)
  print('Original shape X : {} Y: {}'.format(X.shape, Y.shape))

  X_s_train, Y_s_train, X_s_test, Y_s_test, tags_dict, words_dict, test_doc_ids = get_absolute_train_data(X, Y)
  print('absolute train (after filtering) shape X : {} Y: {}'.format(X_s_train.shape, Y_s_train.shape))
  print('absolute test (after filtering) shape X : {} Y: {}'.format(X_s_test.shape, Y_s_test.shape))

  # additional filtering in order to keep the shape same for our training and test Y (tags) data, we are taking
  # the set of tags of both the train and test data , and calculate the intersection of these sets. And then
  # remove any tags from training / test data if that doesnt lies in the intersection. This step is done to keep
  # the shape of Y_train & Y_test same, so that we can predict the tags later from our MultiLabelBinarizer using inverseTransform

  y_train_Set = set()
  for y_row in Y_s_train:
    for t in y_row:
      y_train_Set.add(t.strip(' '))

  y_test_Set = set()
  for y_row in Y_s_test:
    for t in y_row:
      y_test_Set.add(t.strip(' '))

  print('-' * 50)
  print('Y_train_set : {}'.format(len(y_train_Set)))
  print('Y_test_set len : {}'.format(len(y_test_Set)))
  intersection = y_train_Set.intersection(y_test_Set)
  print('Y intersectoin len : {}'.format(len(intersection)))
  f = open('intersection', 'w')
  for t in sorted(list(intersection)):
    f.write('{}\n'.format(t))
  f.close()
  print('-' * 50)


  # cleaning testing set. dropping tags that are not in the intersection
  ft_y_trian = set()
  empty_Y_train = [None] * Y_s_train.shape[0]
  for index in range(len(Y_s_train)):
    filtered_row = list()
    # print('b4 : {}'.format(Y_s_train[index]))
    for tag in range(len(Y_s_train[index])):
      tag_val = Y_s_train[index][tag]
      if ((tag_val in intersection) == True):
        filtered_row.append(tag_val)
        ft_y_trian.add(tag_val)
    empty_Y_train[index] = filtered_row
    # print('af : {}'.format(empty_Y_train[index]))

  cleaned_Y_Train = np.asarray(empty_Y_train)

  print('-'*50)
  print('old without cleaning trssn: {}'.format(Y_s_train.shape))
  print('Aftesr cleaning trssn : {}'.format(cleaned_Y_Train.shape))

  # cleaning testing set. dropping tags that are not in the intersection
  ft_y_test = set()
  empty_Y_test = [None] * Y_s_test.shape[0]
  for index in range(len(Y_s_test)):
    filtered_row = list()
    for tag in range(len(Y_s_test[index])):
      tag_val = Y_s_test[index][tag]
      if ((tag_val in intersection) == True):
        filtered_row.append(tag_val)
        ft_y_test.add(tag_val)
    empty_Y_test[index] = filtered_row

  cleaned_Y_Test = np.asarray(empty_Y_test)

  print('-' * 50)
  print('old without cleaning test : {}'.format(Y_s_train.shape))
  print('Aftesr cleaning test: {}'.format(cleaned_Y_Test.shape))


  print('ft_y_train  size ; {}'.format(len(ft_y_trian)))
  print('ft_y_test  size ; {}'.format(len(ft_y_test)))
  print('are quql : {}'.format(ft_y_test == ft_y_trian))

  print('-'*50)
  print('X_train shape : {}'.format(X_s_train.shape))
  print('Y_train shape : {}'.format(Y_s_train.shape))
  print('X_test shape : {}'.format(X_s_test.shape))
  print('Y_test shape : {}'.format(Y_s_test.shape))
  print('words count : {}'.format(len(words_dict)))
  print('tags count : {}'.format(len(tags_dict)))
  uniq_tags_count = len(tags_dict)
  print('Unique tags : {}'.format(uniq_tags_count))
  print('-'*50)

  # calling the classifier
  oneVsRest_Pipeline(X_s_train, X_s_test, cleaned_Y_Train, cleaned_Y_Test, X_test_2P, words_dict, data_files, test_doc_ids)


  # Trying LogReg Classifier
  # oneVsRest_LogReg(X_train, X_test, Y_train, Y_test, word_dict, tags_dict)

  # Trying SVC linear Classfier
  # oneVsRest_SVCLinear(X_train, X_test, Y_train, Y_test, word_dict, tags_dict)

  # Trying LogReg Classifier o TF_IDF transaformed data
  # oneVsRest_LogReg_TfIdf(X_s_train, X_s_test, cleaned_Y_Train, cleaned_Y_Test, words_dict, tags_dict, data_files, test_doc_ids)

if __name__ == '__main__':
  main()
