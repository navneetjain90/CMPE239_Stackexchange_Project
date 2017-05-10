# Import Statements

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.adapt import mlknn
from sklearn import preprocessing
import numpy as np
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD



# Global Variables

dataSetWords=set()
dataSetLabels=set()
tagsData=[]
trainData=[]
testdata=[]
outputList=[]
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

# Function Definition

def replace_special_character(document):
    r"""This function will remove all the special characters from the doc. This is a preprocessing step"""
    result = re.sub('[^a-zA-Z\n\.]', ' ', document).replace('.', ' ')
    result = ' '.join(result.split())
    result = "".join(result.splitlines())
    result = re.sub(r'\b\w{1,3}\b', '', result)
    return result.strip()

def removestopword(document):
    r"""This function removes the stop words from the dataset basically reduces the dimensionality"""
    text = ' '.join([word for word in document.strip().lower().split() if word not in cachedStopWords])
    return text

def removehtmlTags(input):
    r"""This function removes the HTML tags from the data and converts it into plain text format"""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', input)
    return cleantext

def filterLen(document, minlen):
    r"""This function filters the words with small length as that are not significant in tags classification"""
    text=' '.join(word for word in document.split() if len(word)>=minlen)
    return text

def readTrainingDataSet():
    r"""This function reads the data from the input CSV files and convert it into a data frame"""
    files = ['./data/biology.csv','./data/cooking.csv','./data/crypto.csv','./data/diy.csv','./data/robotics.csv','./data/travel.csv']
    #files = ['./data/cooking_short.csv','./data/crypto_short.csv']
    train_data_frames = []
    for f in files:
        train_data_frames.append(createData(f))
    print('Data inside Dataframe for files {}'.format(files))
    return pd.concat(train_data_frames)

def createData(fileName):
    r"""This function will take the training data files as input parameter and convert it into the data frame consisting of text and 
    tags. Further in this function we will remove the HTML tags, special characters, stopwords from the data set and return a data frame."""
    data_frame = pd.read_csv(fileName, names=['title', 'content', 'tags'])
    data_frame['title'] = data_frame['title'].apply(lambda x: removehtmlTags(x))
    data_frame['content'] = data_frame['content'].apply(lambda x: removehtmlTags(x))
    data_frame['text'] = data_frame[['title', 'content']].apply(lambda x: ''.join(x), axis=1)
    #data_frame['text'] = data_frame['text'].apply(lambda x: removestopword(x))
    data_frame['text'] = data_frame['text'].apply(lambda x: replace_special_character(x))
    data_frame['text'] = data_frame['text'].apply(lambda x: filterLen(x,4))
    #data_frame['tags'] = data_frame['tags'].apply(lambda x: filterLen(x, 3))
    data_frame.drop('title', 1, inplace= True)
    data_frame.drop('content', 1, inplace= True)
    return data_frame



def createTestData(fileName):
    r"""This function will take test file as input parameter and convert it into the data frame consisting of text and 
    tags. Further in this function we will remove the HTML tags, special characters  from the data set."""
    data_frame = pd.read_csv(fileName, names=['title', 'content', 'tags'])
    data_frame['title'] = data_frame['title'].apply(lambda x: removehtmlTags(x))
    data_frame['content'] = data_frame['content'].apply(lambda x: removehtmlTags(x))
    data_frame['text'] = data_frame[['title', 'content']].apply(lambda x: ''.join(x), axis=1)
    #data_frame['text'] = data_frame['text'].apply(lambda x: removestopword(x))
    data_frame['text'] = data_frame['text'].apply(lambda x: replace_special_character(x))
    data_frame['text'] = data_frame['text'].apply(lambda x: filterLen(x,4))
    data_frame.drop('title', 1, inplace= True)
    data_frame.drop('content', 1, inplace= True)
    return data_frame


def createWordSet(document):
    r"""This function takes documents as input and return a unique set of words from the document."""
    for w in document:
        doc = w.split()
        for x in doc:
            if x not in dataSetWords:
                dataSetWords.add(x)
    return


def createTagSet(document):
    r"""This function will take tag documents as input and return a unique set of tags which can be used tp train the 
    alorithm."""
    for w in document:
        doc = w.split()
        for x in doc:
            if x not in dataSetLabels:
                dataSetLabels.add(x)
    return

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
    r"""This function takes array as input and converts it into a list of lists"""
    list_in=[]
    for d in doc:
        docu=d.split()
        list3=[]
        for w in docu:
            list3.append(w)
        list_in.append(list3)
    return list_in

def writeToFile(inputRow):
    r"""This function writes the output to the file for comparison."""
    with open("format.dat", "w") as log:
        for x in inputRow:
            for word in x:
                log.write(word)
                log.write(" ")
            log.write('\n')

def svd_dr(X):
    r"""This is dimensionality reduction method which uses the TruncatedSVD approach to reduce the dimensions."""
    print('Old Shape : {}'.format(X.shape))
    svd = TruncatedSVD(n_components=35000, n_iter=5, random_state=42)
    # svd = PCA(copy=True, iterated_power='auto', n_components=1000, random_state=None,svd_solver='full', tol=0.0, whiten=False)
    svd.fit(X)
    print(svd.explained_variance_ratio_)
    X_new = svd.fit_transform(X)
    print('New Shape : {}'.format(X_new.shape))
    return X_new

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
# Main function

def main():
    r"""Read the data from the files and convert it into data frame and preprocess the data for removal of
    unwanted data"""
    train_data_frame = readTrainingDataSet()
    test_data_frame = createTestData('./data/biology.csv')
    tagsData = splitData(train_data_frame['tags'])
    createWordSet(test_data_frame['text'])
    createWordSet(train_data_frame['text'])
    createTagSet(train_data_frame['tags'])

    #Create CSR matrix for the train data and the text data. Used TfidfVectorizer to create a CSR matrix for test and
    #train data.

    tf = TfidfVectorizer(norm='l2', vocabulary=list(dataSetWords))
    #tf = TfidfVectorizer(vocabulary=list(dataSetWords))
    training_M = tf.fit_transform(train_data_frame["text"])
    testing_M = tf.fit_transform(test_data_frame["text"])

    #Create Binary CSR Matrix for the training tags
    mlb = preprocessing.MultiLabelBinarizer(classes=list(dataSetLabels))
    Y_train = mlb.fit_transform(tagsData)
    csr_info(Y_train)
    csr_info(training_M)
    csr_info(testing_M)

    #Using MLKNN method to Classify the tags. It will take 10 nearest neighbours into consideration while classifying
    #the tags
    mlk = mlknn.MLkNN(k=10, s=0.0, ignore_first_neighbours=0)

    mlk.fit(training_M, Y_train)
    output = mlk.predict(testing_M)  #Output will be a sparse binary matrix

    csr_info(output)

    out = np.array(output.todense())  # Convert the output into dense matrix to read the data

    rows = out.shape[0]
    cols = out.shape[1]
    print(rows)
    print(cols)
    count = 0
    datalistLabels=list(dataSetLabels)

    for x in range(0, rows):
        temp=[]
        for y in range(0, cols):
            if out[x, y] != 0:
                count += 1
                print(datalistLabels[y])
                temp.append(datalistLabels[y])
        outputList.append(temp)
    writeToFile(outputList)   #Write the tags into a file
    print(count)



if __name__ == '__main__':
  main()