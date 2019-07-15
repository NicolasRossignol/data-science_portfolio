# coding: utf-8

## required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
    
def plot_bar(dat,cluster_labels, barwidth=0.1):
    '''Plotting function
    Draws the profiles of the customer clusters'''
    ncluster = pd.Series(cluster_labels['Gpe']).value_counts()
    lcluster = cluster_labels.sort_values('Gpe').drop_duplicates()['label_names']
    plt.figure(figsize = (dat.shape[1]*2,5))
    
    for i in range(dat.shape[1]):
        bars = dat.iloc[:,i]
        rx = np.arange(len(bars))+i*barwidth
        plt.bar(rx,bars, width=barwidth, edgecolor='black')
    for j in ncluster.index:
        plt.text(j+0.5,-1.1,lcluster.iloc[j])        
        plt.text(j+0.5,-1.3,'n='+str(ncluster[j]))
    plt.xlim(0,dat.shape[0]+1)
    plt.ylim(-1.5,1.5)
    plt.legend(dat.columns, loc='upper right')
    plt.show()

def silh_score(dat,cmin=2,cmax=10):
    '''Computes a Kmeans for a range of clusters [cmin, cmax]
    returns 2 lists of scores: loss and silhouette'''
    loss = []
    silh = []
    for k in range(cmin, cmax):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(dat)
        loss.append(kmeans.score(dat))
        silh.append(silhouette_score(dat, kmeans.labels_))
    return(loss,silh)

def multi_score(dat,cmin=2,cmax=10):
    '''Runs function silh_score for different sizes of the dataset
    ie for a range of two to 8 dimensions (columns)
    returns loss and silhouette scores in 2-dim arrays (nb clusters / nb dimensions)'''
    loss = []
    silh = []
    for i in np.arange(2,9):
        tmploss, tmpsilh = silh_score(dat[:,:i],cmin,cmax)
        loss.extend([tmploss])
        silh.extend([tmpsilh])
    loss= np.array(loss)
    silh = np.array(silh)
    return(loss, silh)

def plot_confusion_matrix(cm, classes, normalize=False,title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Function from Sklearn User Guide
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

 
### returns quartiles 1,2,3,4 (1 being the quartile with 'best' customers)
## Recency: low values are desirable
def RScore(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4

## Frequency & Monetary: high values are desirable
def FMScore(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1