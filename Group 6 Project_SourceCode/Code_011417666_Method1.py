import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn import preprocessing, linear_model
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier





dataSetWords=set()
dataSetLabels=set()
tagsData=[]
trainData=[]
testdata=[]

stop_words = ["", " ", "a", "about", "above", "after", "again", "against", "all", "am", "an",
              "and","any", "are", "aren't", "at", "be", "because", "been", "before", "being",
              "below", "between", "both", "but", "by", "can", "can't",  "cannot", "could",
              "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
              "during", "each", "few", "for", "from", "further", "had",  "hadn't","has",
              "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
              "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
              "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
              "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
              "myself", "no", "nor", "not" , "of", "off", "on", "once", "only", "or", "other",
              "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
              "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
              "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
              "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
              "they've", "this", "those", "to", "too", "under", "until", "up", "very", "was",
              "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
              "what's", "when", "when's", "where", "where's", "which", "while", "who",
              "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't","you",
              "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
              "yourselves", "return", "arent", "cant", "couldnt", "didnt", "doesnt", "dont",
              "hadnt", "hasnt", "havent", "hes", "heres", "hows", "im", "isnt", "its", "lets",
              "mustnt", "shant", "shes", "shouldnt", "thats", "theres", "theyll", "theyre",
              "theyve", "wasnt", "were", "werent", "whats", "whens", "wheres", "whos", "whys",
              "wont", "wouldnt", "youd", "youll", "youre", "youve"]

cachedStopWords = stopwords.words("english")
cachedStopWords.append(stop_words)


def replace_special_character(document):
    result = re.sub('[^a-zA-Z\n\.]', ' ', document).replace('.', ' ')
    result = ' '.join(result.split())
    result = "".join(result.splitlines())
    result = re.sub(r'\b\w{1,3}\b', '', result)
    return result.strip()

def removestopword(document):
    text = ' '.join([word for word in document.strip().lower().split() if word not in cachedStopWords])
    return text

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def filterLen(document, minlen):
    text=' '.join(word for word in document.split() if len(word)>=minlen)
    return text

def createData(fileName):
    data_frame = pd.read_csv(fileName, names=['title', 'content', 'tags']) #,nrows=500)
    data_frame['title'] = data_frame['title'].apply(lambda x: cleanhtml(x))
    data_frame['content'] = data_frame['content'].apply(lambda x: cleanhtml(x))
    data_frame['text'] = data_frame[['title', 'content']].apply(lambda x: ''.join(x), axis=1)
    data_frame['text'] = data_frame['text'].apply(lambda x: removestopword(x))
    data_frame['text'] = data_frame['text'].apply(lambda x: replace_special_character(x))
    #data_frame['text'] = data_frame['text'].apply(lambda x: filterLen(x,4))
    data_frame['tags'] = data_frame['tags'].apply(lambda x: filterLen(x, 3))

    data_frame.drop('title', 1, inplace= True)
    data_frame.drop('content', 1, inplace= True)
    return data_frame

def createTestData(fileName):
    data_frame = pd.read_csv(fileName, names=['title', 'content', 'tags'])#,nrows=100)
    data_frame['title'] = data_frame['title'].apply(lambda x: cleanhtml(x))
    data_frame['content'] = data_frame['content'].apply(lambda x: cleanhtml(x))
    data_frame['text'] = data_frame[['title', 'content']].apply(lambda x: ''.join(x), axis=1)
    data_frame['text'] = data_frame['text'].apply(lambda x: removestopword(x))
    data_frame['text'] = data_frame['text'].apply(lambda x: replace_special_character(x))
    data_frame['text'] = data_frame['text'].apply(lambda x: filterLen(x,3))

    data_frame.drop('title', 1, inplace= True)
    data_frame.drop('content', 1, inplace= True)
    return data_frame


def createWordSet(document):
    for w in document:
        doc = w.split()
        for x in doc:
            if x not in dataSetWords:
                dataSetWords.add(x)
    return

def createTagSet(document):
    for w in document:
        doc = w.split()
        for x in doc:
            if x not in dataSetLabels:
                dataSetLabels.add(x)
    return


def readTrainingDataSet():
    files = [ './data/crypto.csv',
              './data/cooking.csv',
              #'./data/diy.csv',
              #'./data/biology.csv',
              './data/robotics.csv',
              './data/travel.csv']
    train_data_frames = []
    for f in files:
        train_data_frames.append(createData(f))
    print('Data inside Dataframe for files {}'.format(files))
    return pd.concat(train_data_frames)


def csr_info(mat, name="", non_empy=False):
    r""" Print out info about this CSR matrix. If non_empy, 
    report number of non-empty rows and cols as well
    """
    if non_empy:
        print("%s [nrows %d (%d non-empty), ncols %d (%d non-empty), nnz %d]" % (
                name, mat.shape[0],
                sum(1 if mat.indptr[i+1] > mat.indptr[i] else 0
                for i in range(mat.shape[0])),
                mat.shape[1], len(np.unique(mat.indices)),
                len(mat.data)))
    else:
        print( "%s [nrows %d, ncols %d, nnz %d]" % (name,
                mat.shape[0], mat.shape[1], len(mat.data)) )




def splitData(doc):
    list_in=[]
    for d in doc:
        docu=d.split()
        list3=[]
        for w in docu:
            list3.append(w)
        list_in.append(list3)
    return list_in


# classifier = OneVsRestClassifier(LogisticRegression(penalty='l2', C=0.01))

def compareOutput(file1,file2):
    r"""This function takes 2 files as input i.e. input file with correct tags and output file with tags predicted from out function.
    The output will be the percentage of the matches."""
    input=open(file1,"r")
    test=open(file2,"r")
    inputList = []
    testList = []
    posCount=0
    for line in input:
        inputList.append(replace_special_character(line))
    for line in test:
        testList.append(replace_special_character(line))
    totalCount=len(inputList)
    for x in range(totalCount):
        set = False
        doc=testList[x].split()
        for word in doc:
            if word in inputList[x].split():
                if set is False:
                    posCount += 1
                    set=True
    percentage=(posCount/totalCount) * 100
    return percentage



def OVR_Classify(X_train, X_test, Y_train, word_dict, tags_dict, test_tags=None):
    print('Processing : OVR_Classify')
    print('-' * 50)
    from sklearn.feature_extraction.text import TfidfTransformer
    vectorizer = CountVectorizer(min_df=1, vocabulary=word_dict)
    X_v_train = vectorizer.fit_transform(X_train)
    X_v_test = vectorizer.fit_transform(X_test)

    transformer = TfidfTransformer(smooth_idf=False)
    X_train_tf = transformer.fit_transform(X_v_train)
    X_test_tf = transformer.fit_transform(X_v_test)


    #uniq_tags_names = list(tags_dict.keys())
    mlb = preprocessing.MultiLabelBinarizer(classes=list(tags_dict))
    train_model = mlb.fit_transform(Y_train)
    classifier = OneVsRestClassifier(Perceptron(#loss='hinge',
                                                alpha=1e-3,
                                                penalty='elasticnet',
                                                random_state=999,
                                                #class_weight="balanced",
                                                n_iter=50,
                                                #learning_rate='optimal'
    ))

    classifier.fit(X_train_tf, train_model)
    print('-' * 50)
    #print('Score oneVsRest_SGDC_TfIdf : {}'.format(score))
    print('-' * 50)
    Y_pred = classifier.predict(X_test_tf)
    print(Y_pred)
    Y_back = mlb.inverse_transform(Y_pred)
    print(Y_back)

    #print 'score ',classifier.score(X_test_tf, mlb.fit_transform(test_tags))
    #write_to_file(Y_original, Y_back, 'oneVsRest_SGDC_TfIdf', score, data_files)






train_data_frame = readTrainingDataSet()

test_data_frame = createTestData('./data/test.csv')

print(len(train_data_frame['text']))
print(len(train_data_frame['tags']))
print("***********")

tagsData=splitData(train_data_frame['tags'])
trainData=splitData(train_data_frame['text'])
testdata=splitData(test_data_frame['text'])

print(len(tagsData))
print(tagsData[1])


createWordSet(test_data_frame['text'])
createWordSet(train_data_frame['text'])
createTagSet(train_data_frame['tags'])

OVR_Classify(train_data_frame["text"],test_data_frame["text"],tagsData,dataSetWords,dataSetLabels)


