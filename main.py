import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA

###### global vars here ######

cachedStopWords = stopwords.words("english")
word_dict = {}
word_freq = {}

tag_dict = {}
tag_freq = {}

train_data_frame = None
test_data_frame = None

col_array = []
val_array = []
row_array = []

tags_col_array = []
tags_row_array = []
tags_val_array = []

######  global vars ends here ######

def readTestFile():
  file = './data/test.csv'
  data_frame = pd.read_csv(file, names=['id', 'title', 'content'])
  print('finished reading files ... ')
  data_frame['title'] = data_frame['title'].apply(lambda x : removestopword(x))
  data_frame['title'] = data_frame['title'].apply(lambda x : replace_special_character(x))
  data_frame['content'] = data_frame['content'].apply(lambda x: removestopword(x))
  data_frame['content'] = data_frame['content'].apply(lambda x: replace_special_character(x))
  print('finished cleaning...')
  return data_frame

def replace_special_character(document):
    result = re.sub('[^a-zA-Z\n\.]', ' ', document).replace('.', ' ')
    result = ' '.join(result.split())
    result = "".join(result.splitlines())
    result=re.sub(r'\b\w{1,3}\b', '', result)
    return result.strip()

def removestopword(document):
  text = ' '.join([word for word in document.strip().lower().split() if word not in cachedStopWords])
  return text

def readTrain(file):
  print('reading file {}'.format(file))
  data_frame = pd.read_csv(file, names = ['id','title','content','tags'])
  data_frame['title'] = data_frame['title'].apply(lambda x : x.decode('utf-8'))
  data_frame['content'] = data_frame['content'].apply(lambda x : x.decode('utf-8'))
  data_frame['text'] = data_frame[['title', 'content']].apply(lambda x : ''.join(x), axis = 1)
  data_frame['text'] = data_frame['text'].apply(lambda x : removestopword(x))
  data_frame['text'] = data_frame['text'].apply(lambda x : replace_special_character(x))
  # data_frame['content'] = data_frame['content'].apply(lambda x: removestopword(x))
  # data_frame['content'] = data_frame['content'].apply(lambda x: replace_special_character(x))
  data_frame['tags'] = data_frame['tags'].apply(lambda x : x.split(' '))
  data_frame.drop('title', 1, inplace= True)
  data_frame.drop('content', 1, inplace= True)
  data_frame.drop('id', 1, inplace= True)
  print('finished processing file : {}'.format(file))
  return data_frame

def readTrainingDataSet():
  files = ['./data/biology.csv','./data/cooking.csv','./data/crypto.csv','./data/diy.csv','./data/robotics.csv','./data/travel.csv']
  #files = ['./data/cooking_short.csv','./data/crypto_short.csv']
  train_data_frames = []
  for f in files:
    train_data_frames.append(readTrain(f))
  print('Data inside Dataframe for files {}'.format(files))
  return pd.concat(train_data_frames)

def assign_ids(document):
  doc = document.split()
  for w in doc:
    if w not in word_dict:  # assigning ID to the words
      word_dict[w] = len(word_dict)
    if w not in word_freq:  # creating the frequency count for the words
      word_freq[w] = 1
    else:
      word_freq[w] += 1
  return

def assign_tag_ids(document):
  for w in document:
    if w not in tag_dict:  # assigning ID to the words
      tag_dict[w] = len(tag_dict)
    if w not in tag_freq:  # creating the frequency count for the words
      tag_freq[w] = 1
    else:
      tag_freq[w] += 1
  return

def head_dict(dict, n, title):
  print('============ {} ============ '.format(title))
  for key in dict:
    if(n == 0):
      break
    print('{} , {}'.format(key, dict[key]))
    n -=1
  print('============ /{}/ ============ '.format(title))

def create_feature_ids(data_frame):
  data_frame['text'].apply(lambda x: assign_ids(x))
  data_frame['tags'].apply(lambda x: assign_tag_ids(x))

def create_coo_data(data_frame):
  row_id = 0;
  for index, row in data_frame.iterrows():
    doc = row['text'].split()
    doc_map = {}
    for w in doc:
      if w not in doc_map:
        doc_map[w] = 1
      else:
        doc_map[w] += 1
    for w in set(doc):
      col_array.append(word_dict[w])
      row_array.append(row_id)
      val_array.append(doc_map[w])
    row_id += 1

  # creating coo data for tags
  row_id = 0;
  for index, row in data_frame.iterrows():
    tag_doc = row['tags']
    tag_doc_map = {}
    for w in tag_doc:
      if w not in tag_doc_map:
        tag_doc_map[w] = 1
      else:
        tag_doc_map[w] += 1
    for w in set(tag_doc):
      tags_col_array.append(tag_dict[w])
      tags_row_array.append(row_id)
      tags_val_array.append(tag_doc_map[w])
    row_id += 1
  return

def create_csr_matrix(coo_cols_array,coo_rows_array,coo_vals_array, rows, cols):
  csr_matrix = coo_matrix((coo_vals_array,(coo_rows_array, coo_cols_array)), shape = (rows, cols)).tocsr()
  csr_matrix
  return csr_matrix

from sklearn.decomposition import TruncatedSVD
def svd_dr(X):
  print('Old Shape : {}'.format(X.shape))
  svd = TruncatedSVD(n_components=10000, n_iter=7, random_state=42)
  svd.fit(X)
  print(svd.explained_variance_ratio_)
  X_new = svd.fit_transform(X)
  print('New Shape : {}'.format(X_new.shape))
  return X_new

def main():
  train_data_frame = readTrainingDataSet()
  print(train_data_frame.head())
  print('No of documents, features : {}'.format(train_data_frame.shape))
  create_feature_ids(train_data_frame)
  print('Number of unique words : {}'.format(len(word_dict)))
  print('Number of unique tags : {}'.format(len(tag_dict)))
  head_dict(tag_dict, 5,'tag Dictionary')
  head_dict(tag_freq, 5,'tag Freq Dictionary')

  create_coo_data(train_data_frame)
  coo_col_array = np.asarray(col_array)
  coo_row_array = np.asarray(row_array)
  coo_val_array = np.asarray(val_array)
  totalRows = train_data_frame.shape[0]
  totalFeatures = len(word_dict)
  text_csr_matrix = create_csr_matrix(coo_col_array, coo_row_array, coo_val_array, totalRows, totalFeatures)
  print('=========================Csr Matrix=========================')
  print('{}'.format(text_csr_matrix ))

  tag_coo_col_array = np.asarray(tags_col_array)
  tag_coo_row_array = np.asarray(tags_row_array)
  tag_coo_val_array = np.asarray(tags_val_array)
  tag_totalRows = train_data_frame.shape[0]
  tag_totalFeatures = len(tag_dict)
  tag_csr_matrix = create_csr_matrix(tag_coo_col_array, tag_coo_row_array, tag_coo_val_array, tag_totalRows, tag_totalFeatures)
  print('=========================TAGS Csr Matrix=========================')
  print('{}'.format(tag_csr_matrix))
  reduced_csr = svd_dr(text_csr_matrix)

  print('=========================Reduced Csr Matrix=========================')
  print('{}'.format(reduced_csr))

  # # test = readTestFile()
  # print('=========================TEST=========================')
  # count = 0
  # for index, row in test.iterrows():
  #   if (count == 3):
  #     break
  #   print(row['title'])
  #   count += 1

if __name__ == '__main__':
  main()
