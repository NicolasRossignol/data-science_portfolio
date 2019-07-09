import pandas as pd
import numpy as np
    
def df_normalization(df):
    '''Makes the norm of the vectors (lines)
    equal to 1'''
    norm_df = df.mul(df).sum(axis=1)
    return df.mul(1/np.sqrt(norm_df),axis=0)

def col_weighting(df, fun=None):
    '''applies a weighting on columns
    based on the marginal sum of columns
    default: multiply by marginal sum of columns
    any simple function can be applied on the marginal sum
    before if provided
    example: fun=np.sqrt or fun=lambda x: 1/x'''
    ### marginal sum of columns
    wei = df.sum(axis=0)
    ### optional: function to be applied to the sum wei
    if fun is not None:
        wei = fun(wei)
    return df.mul(wei, axis=1)

def chi_tab(df1):
    '''adapted from function decostand
    from R package vegan
    After applying this transformation to a matrix,
    applying euclidean distances should yield chi-square distances'''
    # chi.square: divide by row sums and square root of column sums, 
    # and adjust for square root of matrix total 
    output = df1.mul(1/np.sqrt(df1.sum(axis=0))).mul(
    1/df1.sum(axis=1),axis=0)*np.sqrt(df1.sum().sum())
    return output

def extract_cat(dat, token_pattern='(?u)\b\w\w+\b', verbose=True):
    from sklearn.feature_extraction.text import CountVectorizer
    vect = CountVectorizer(token_pattern=token_pattern)
    vect.fit(dat)
    col_names = list(vect.vocabulary_.keys())
    col_names.sort()
    df = pd.DataFrame(vect.transform(dat).toarray(),columns=col_names)
    if verbose:
        print('length vocabulary: ', len(vect.vocabulary_))
        print('vocabulary: \n',vect.vocabulary_)
    return df

