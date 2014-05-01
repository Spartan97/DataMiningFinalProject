# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 19:20:08 2014

@author: dartslab
"""
import matplotlib.pyplot as pl
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn import linear_model
from sklearn import neighbors, datasets
from sklearn.neighbors import NearestNeighbors
from sklearn import tree
from sklearn.cross_validation import cross_val_score
import pandas as pd
import random
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import threading

def pairs(data):
    """Generates and shows a pairwise scatterplot of the dataset features.

    A figure with nxn scatterplots is generated, where n is the number of features. The features are
    defined as the all columns excluding the final column, which is defined as the class.

    Args:
      data (array): A dataset.

    """
    i = 1

    # Divide columns into features and class
    features = list(data.columns)
    classes = features[-1] # create class column
    del features[-1] # delete class column from feature vector

    # Generate an nxn subplot figure, where n is the number of features
    figure = pl.figure(figsize=(5*(len(data.columns)-1), 4*(len(data.columns)-1)))
    for col1 in data[features]:
        for col2 in data[features]:
            ax = pl.subplot(len(data.columns)-1, len(data.columns)-1, i)
            if col1 == col2:
                ax.text(2.5, 4.5, col1, style='normal', fontsize=20)
                ax.axis([0, 10, 0, 10])
                pl.xticks([]), pl.yticks([])
            else:
                for name in data[classes]:
                    cond = data[classes] == name
                    ax.plot(data[col2][cond], data[col1][cond], linestyle='none', marker='o', label=name)
                #t = plt.title(name)
            i += 1

    pl.show()
    
def normalize(df):
    #normalizes the values from 0-100    
    for i in df.columns:
        print "I:"+str(i)
        mmin = df[i].min()
        mmax = df[i].max()
        for j in df[i].index:
#            print "J:"+str(j)
            df[i][j]=(df[i][j]- mmin)/(mmax-mmin)*100.0
    return df



def KNNThread(cols):
    if type(cols) == type(1):
        X = small_df[[cols]]
    else:
        X = small_df[cols]
    results = str(cols)   
    
    neigh = neighbors.KNeighborsClassifier(3, 'distance')
    scores = cross_val_score(neigh, X, Y, cv=3, scoring='accuracy')
    results += "\t3 Nearest Neighbors\t" +str(sum(scores)/float(len(scores)))    
    print results  
    
    f = open('results.txt','a')
    f.write(results)
    f.write('\n')
    f.close() 

def CVThread(cols):
    if type(cols) == type(1):
        X = small_df[[cols]]
    else:
        X = small_df[cols]
    results = str(cols)   
    
    svc = svm.SVC()
    scores = cross_val_score(svc, X, Y, cv=3, scoring='accuracy')
    results += "\tSupport Vector Classification\t" +str(sum(scores)/float(len(scores)))    
    print results  
    
    f = open('results.txt','a')
    f.write(results)
    f.write('\n')
    f.close() 
    
def GaussThread( cols):
    if type(cols) == type(1):
        X = small_df[[cols]]
    else:
        X = small_df[cols]
    results = str(cols)   
    
    gnb = GaussianNB()    
    scores = cross_val_score(gnb, X, Y, cv=3, scoring='accuracy')
    results += "\tGaussian Naive Bayes\t" +str(sum(scores)/float(len(scores)))    
    print results  
    
    f = open('results.txt','a')
    f.write(results)
    f.write('\n')
    f.close() 

def DecTreeThread( cols):
    if type(cols) == type(1):
        X = small_df[[cols]]
    else:
        X = small_df[cols]
    results = str(cols)   
    
    dec = tree.DecisionTreeClassifier()
    scores = cross_val_score(dec, X, Y, cv=3, scoring='accuracy')
    results += "\tDecision Tree\t" +str(sum(scores)/float(len(scores)))    
    print results  
    
    f = open('results.txt','a')
    f.write(results)
    f.write('\n')
    f.close() 
    
    
torrents = []
for t_id in all_torrents:
    t = all_torrents[t_id]
    bad = 0
    if t.downvotes>t.upvotes:
        bad = 1
    torrentInfo = [t.id,t.title,t.magnet,t.size,t.seeders,t.leechers,t.upvotes,t.downvotes,t.uploaded,t.nfo, bad]
    torrents.append(torrentInfo)
df = pd.DataFrame(torrents)
df = df.dropna()

#rows = random.sample(df.index, 100)
#small_df = df.ix[rows]
#small_df = small_df[[3,4,5,6,7,10]]
#small_df = normalize(small_df)
#pairs(small_df)

rows = random.sample(df.index, 50000)
small_df = df.ix[rows]
small_df = small_df[[3,4,5,6,7,10]]
Y = small_df[10] 

#tests = [3, 4, 5, 6, 7, [4,5]]
tests = [[3,4],[3,5], [3,4,5]]

threads = []
for cols in tests:
    print cols
    a = threading.Thread(target=KNNThread, args = (cols,))   
    b = threading.Thread(target=CVThread, args = (cols,))   
    c = threading.Thread(target=GaussThread, args = (cols,))   
    d = threading.Thread(target=DecTreeThread, args = (cols,))   
    
    threads.append(a)
    threads.append(b)
    threads.append(c)
    threads.append(d)

for thread in threads:
    thread.daemon = True
    thread.start()
print 'all threads launched...'
for thread in threads:
    thread.join()
print 'all done!'
    
    
    
    
