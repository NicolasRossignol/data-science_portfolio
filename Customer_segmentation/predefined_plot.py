# coding: utf-8

## required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_article_pca(pca, result, ArticleTable):
    """predefined plot for ArticleTable data"""
    pcs = pca.components_
    tmp = pd.DataFrame(pcs.T)
    p_lab = ArticleTable.columns
    
    fig = plt.figure(figsize=(15,10))
    plt.subplot(2,2,1)
    for i, (x,y) in enumerate(zip(pcs[0,:], pcs[1,:])):
        plt.plot([0,x], [0,y], color='k')
        plt.text(x, y, p_lab[i],fontsize='10')
    plt.plot([-1,1],[0,0], color='grey', ls='--')
    plt.plot([0,0],[-1,1], color='grey', ls='--')  
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')   

    plt.subplot(2,2,2)
    for i, (x,y) in enumerate(zip(pcs[1,:], pcs[2,:])):
        plt.plot([0,x], [0,y], color='k')
        plt.text(x, y, p_lab[i],fontsize='10')
    plt.plot([-1,1],[0,0], color='grey', ls='--')
    plt.plot([0,0],[-1,1], color='grey', ls='--')  
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.xlabel('PC 2')
    plt.ylabel('PC 3')   

    plt.subplot(2,2,3)
    plt.plot(result[:,0],result[:,1],'.')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')   

    plt.subplot(2,2,4)
    plt.plot(result[:,1],result[:,2],'.')
    plt.xlabel('PC 2')
    plt.ylabel('PC 3') 
    plt.show()

def plot_km_scores(loss, silh):
    """predefined plot for Kmeans clusters quality"""
    plt.figure(figsize=(12, 8))
    plt.subplot(1,2,1)
    for i in range(loss.shape[0]):
        plt.plot(np.arange(2,10), loss[i,:], label='PC'+str(i+2))
    plt.title("Change in Loss from additional cluster" )
    plt.ylabel("Loss of K-Means")
    plt.xlabel("Number of Clusters in K-Means")
    plt.grid()
    plt.legend()

    plt.subplot(1,2,2)
    for i in range(silh.shape[0]):
        plt.plot(np.arange(2,10), silh[i,:], label='PC'+str(i+2))
    plt.title("Change in avg Silhouette from additional cluster" )
    plt.ylabel("Avg Silh")
    plt.xlabel("Number of Clusters in K-Means")
    plt.grid()
    plt.legend()
    plt.show()
    
def plot_customer_pca(pca, result, df_cluster_labels5, FinalTable):
    '''Predefined plotting function for
    displaying customer PCA'''
    cl_names=['Wholesal_loyal', 'Wholesal_occas', 'Retail_specific',
              'Retail_tester', 'Retail_general']
    cl_col=['red','blue','orange', 'purple', 'green']
    pcs = pca.components_
    p_lab = FinalTable.columns
    val_min = 0.3

    plt.figure(figsize = (15,15))
    ax1 = plt.subplot(2,2,1)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')  
    ax2 = plt.subplot(2,2,2)
    plt.xlabel('PC 3')
    plt.ylabel('PC 4') 
    for i in range(5):
        idx = np.where(df_cluster_labels5.label_names==cl_names[i])
        ax1.scatter(result[idx,0], result[idx,1], c=cl_col[i], alpha=0.5)
        ax2.scatter(result[idx,2], result[idx,3], c=cl_col[i], alpha=0.5)
    ax1.legend(cl_names)   

    plt.subplot(2,2,3)
    for i, (x,y) in enumerate(zip(pcs[0,:], pcs[1,:])):
        if (abs(x)>=0.3) or (abs(y)>=val_min):
            plt.plot([0,x], [0,y], color='k')
            plt.text(x, y, p_lab[i],fontsize='10')
    plt.plot([-1,1],[0,0], color='grey', ls='--')
    plt.plot([0,0],[-1,1], color='grey', ls='--')  
    plt.xlim([-0.5,0.5])
    plt.ylim([-0.5,0.5])
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')   

    plt.subplot(2,2,4)
    for i, (x,y) in enumerate(zip(pcs[2,:], pcs[3,:])):
        if (abs(x)>=0.3) or (abs(y)>=val_min):
            plt.plot([0,x], [0,y], color='k')
            plt.text(x, y, p_lab[i],fontsize='10')
    plt.plot([-1,1],[0,0], color='grey', ls='--')
    plt.plot([0,0],[-1,1], color='grey', ls='--')  
    plt.xlim([-0.5,0.5])
    plt.ylim([-0.5,0.5])
    plt.xlabel('PC 3')
    plt.ylabel('PC 4')   

    plt.show()