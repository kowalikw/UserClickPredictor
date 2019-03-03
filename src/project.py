# User Click Predictor
# Wojciech Kowalik
# 2019

import numpy as np
import pandas as pd
import scipy.sparse
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.externals.joblib import Memory
from sklearn.metrics import roc_curve, auc


file_training = "../data/D80M.tsv"
file_test = "../data/D5M_test_x.tsv"

memory_cache = Memory("/tmp/mycache")
batch_size = 100000

# read header
with open(file_training) as f:
    header = f.readline().strip().split("\t")
    print(header)
pandas_reader = None

@memory_cache.cache
def load_data(i):
    print("Chunk no ", i)
    try:
        X = pandas_reader.get_chunk(batch_size)
    except:
        X = None
    return X

def read_by_chunks(fname):
    global pandas_reader
    pandas_reader = pd.read_csv(fname, sep='\t', skiprows=1, chunksize=batch_size,
                                names=header)
    i = 0
    while True:
        X = load_data(i)
        i = i + 1
        if X is None:
            break
        yield transform_chunk(X)
    pandas_reader.close()

@memory_cache.cache
def transform_chunk(X):
    # 'small' variables
    y = X.iloc[:, 0].values
    X_DispUrl = X.iloc[:, np.array([1])].values
    X_AdId = X.iloc[:, np.array([2])].values
    X_AdvId = X.iloc[:, np.array([3])].values
    
    #'large' variables
    X_AdKeyWord = X.iloc[:, np.array([9])].values
    X_AdTitle = X.iloc[:, np.array([10])].values
    X_AdDesc = X.iloc[:, np.array([11])].values
    X_Query = X.iloc[:, np.array([12])].values
    X = X.iloc[:, np.array([4,5,7,8])].values    
    
    # keep original values + one hot encoding
    one_hot_encoder = OneHotEncoder(categorical_features=[0,1,2,3], n_values=[3,3,3,6], handle_unknown="ignore")
    feature_hasher = FeatureHasher(n_features=1000000, input_type="string")
    hashing_vectorizer = HashingVectorizer()
    
    # use FeatureHasher
    X_DispUrl_Result = feature_hasher.fit_transform(X_DispUrl.astype(str));
    X_AdId_Result = feature_hasher.fit_transform(X_AdId.astype(str));
    X_AdvId_Result = feature_hasher.fit_transform(X_AdvId.astype(str));
    
    # use HashingVectorizer
    X_AdKeyword_Result = hashing_vectorizer.fit_transform(np.ravel(X_AdKeyWord.tolist()))
    X_AdTitle_Result = hashing_vectorizer.fit_transform(np.ravel(X_AdTitle.tolist()))
    X_AdDesc_Result = hashing_vectorizer.fit_transform(np.ravel(X_AdDesc.tolist()))
    X_Query_Result = hashing_vectorizer.fit_transform(np.ravel(X_Query.tolist()))
    
    # use OneHotEncodel
    X[:,[0,1,3]] -= 1 # some vars indexing starts with 1
    X_ohe = one_hot_encoder.fit_transform(X)
    
    # concatenate all variables
    X = scipy.sparse.hstack([X_ohe, X_DispUrl_Result, X_AdId_Result, X_AdvId_Result, X_AdKeyword_Result, X_AdTitle_Result, X_AdDesc_Result, X_Query_Result])
    
    return X, y

# open training and test file
reader = read_by_chunks(file_training)
readerTest = read_by_chunks(file_test)

# create model
model = SGDClassifier(max_iter=1000, tol=1e-3, loss='log')

# train model
i = 0
for X, y in reader:
    model.partial_fit(X, y, classes=[0,1])
    i = i + 1

# tcompute scores
i = 0
scores = []
results = []
for X, y in readerTest:
    scores.extend(model.decision_function(X))
    results.extend(y)  
    i = i + 1
    
# save scores to a result  file
file_result = open("kowalikw_scores.txt","w") 
file_result.write("WOJCIECH KOWALIK\n") 
for s in scores:
    file_result.write(str(s) + "\n")
file_result.close()