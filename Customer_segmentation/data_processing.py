# coding: utf-8

## Function used to clean and pre-process dataset

## required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

def step_clean(dat):
    """ Step1: data cleaning before further processing"""
    dat = dat[pd.notnull(dat['CustomerID'])]
    dat = dat[dat['Country']=='United Kingdom']
    dat['InvoiceDate'] = pd.to_datetime(dat['InvoiceDate'])
    dat['Gross'] = dat['Quantity'] * dat['UnitPrice']
    
    indices=[]
    indices.extend(dat[dat['StockCode']=='BANK CHARGES'].index)
    indices.extend(dat[dat['StockCode']=='C2'].index)
    indices.extend(dat[dat['StockCode']=='DOT'].index)
    indices.extend(dat[dat['StockCode']=='M'].index)
    indices.extend(dat[dat['StockCode']=='PADS'].index)
    indices.extend(dat[dat['StockCode']=='POST'].index)
    dat = dat.drop(index=indices)
    
    return(dat)

def find_cancelled(dat):
    if str(dat)[0]=='C':
        return(True)
    else:
        return(False)
    
def Mean_Basket(R_dat):
    """ Feature engineering: 'average basket' for 1 Customer"""
    InvoiceTable = R_dat.groupby('InvoiceNo').agg({
        'Gross': 'sum',
        'Quantity': ['sum', lambda x: -sum((x/sum(x)) * np.log(x/sum(x)))],
        'StockCode': lambda x: len(x),
        'UnitPrice': ['min', 'max', 'mean'],
        'CustomerID': lambda x: x.unique(),
        'ARTpop': 'mean',
        'ARTcost': 'mean'
    })
    InvoiceTable.columns = ['MB_TotValue','MB_NbArt','MB_Diversity',
                            'MB_NbUnArt','MB_PUmin','MB_PUmax','MB_PUmean',
                            'CustomerID', 'MB_ARTpop', 'MB_ARTcost']
    InvoiceTable['MB_Evenness'] = InvoiceTable['MB_Diversity']/np.log(InvoiceTable['MB_NbUnArt'])
    ## si division par log(1), l'équitabilité est minimale
    InvoiceTable.loc[InvoiceTable[pd.isnull(InvoiceTable['MB_Evenness'])].index,'MB_Evenness'] = 0
    ## normalement on attend un seul customer, doc une seule ligne aprés aggrégation
    MeanBasketTable = InvoiceTable.groupby('CustomerID').agg('mean')
    return(MeanBasketTable)

def set_rfm(R_dat, NOW = dt.datetime(2011,12,10)):
    """Computes the 3 RFM metrics and returns a table"""
    ## NOW can be changed
    R_dat['InvoiceDate'] = pd.to_datetime(R_dat['InvoiceDate'])
    rfmTable = R_dat.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (NOW.date() - x.max().date()).days,
        'InvoiceNo': [ lambda x: x.nunique()],
        'Gross': lambda x: x.sum()
        })
    rfmTable['InvoiceDate'] = rfmTable['InvoiceDate'].astype(int)
    rfmTable.rename(columns={'InvoiceDate': 'recency', 
                             'InvoiceNo': 'frequency', 
                            'Gross': 'monetary_value'}, inplace=True)
    rfmTable.columns = ['recency','frequency','monetary_value']
    return(rfmTable)

def set_CustomerFeatures(R_dat):
    """Computes summary statistics aggregated by customers and returns a table"""    
    CustomerTable = R_dat.groupby('CustomerID').agg({
        'Quantity': ['sum', lambda x: -sum((x/sum(x)) * np.log(x/sum(x)))],
        'StockCode': lambda x: x.nunique(),
        'UnitPrice': ['min', 'max', 'mean']
        })

    CustomerTable.columns = ['NbArt','Diversity',
                            'NbUnArt','PUmin','PUmax','PUmean']
    CustomerTable['Evenness'] = CustomerTable['Diversity']/np.log(CustomerTable['NbUnArt'])
    CustomerTable.loc[CustomerTable[pd.isnull(CustomerTable['Evenness'])].index,'Evenness'] = 0
    CustomerTable.loc[CustomerTable['Evenness']==np.inf,'Evenness'] = 0
    CustomerTable['BulkPurchase'] = CustomerTable['NbArt']/CustomerTable['NbUnArt']
    
    ##### probability being a wholesaler: 
    # logistic function set to represent my own subjective expertise
    infl_pt = 20 
    k = 0.2 
    CustomerTable['p_wholesalers'] = 1/(1+np.exp(-k*(CustomerTable['BulkPurchase']-infl_pt)))
    
    return(CustomerTable)

def set_CancelledTable(C_dat):
    """Computes summary statistics aggregated by customers 
    for cancelled invoices and return a table"""
    CancelledTable = C_dat.groupby('CustomerID').agg({
    'Quantity': 'sum', 
    'StockCode': lambda x: x.nunique(),
    'Gross': 'sum'
        })
    CancelledTable.rename(columns={
        'Quantity': 'C_NbArt', 
        'StockCode': 'C_NbInvoice', 
        'Gross': 'C_monetary'}, inplace=True)
    return(CancelledTable)
    
def set_dataFeatures(dat):
    """Main program: processes main all feature calculation for customer sequence data
    can accept sequence data with several customers
    return a table with unique customers in row and features in columns
    incorrect value will lead to an empty final table"""
    
    ## if some codes are numeric, must avoid type 'float' and force 'object'
    dat['StockCode'] = dat['StockCode'].astype(dtype= {"StockCode":"object"})
    #1.cleaning
    dat = step_clean(dat)
    #2. Regular vs Cancelled
    cancelled_list = dat['InvoiceNo'].apply(find_cancelled)
    C_dat = dat[cancelled_list]
    R_dat = dat[cancelled_list==False]
    #3. implement 'ARTpop', 'ARTcost'
    ArticleCharacteritics = pd.read_csv('ArticleCharacteritics.csv',
                                    index_col=0)
    liste = R_dat['StockCode'].astype(str)
    tmp = ArticleCharacteritics.loc[liste,:].reset_index()
    R_dat = R_dat.reset_index()
    R_dat.loc[:,'ARTpop'] = tmp.loc[:,'ARTpop'] 
    R_dat.loc[:,'ARTcost'] = tmp.loc[:,'ARTcost']
    R_dat.index = R_dat['index']
    R_dat = R_dat.drop(columns=['index'])
    #4. MeanBasket
    MeanBasketTable = Mean_Basket(R_dat)
    # Final table
    rfmTable = set_rfm(R_dat)
    CustomerTable = set_CustomerFeatures(R_dat)
    CancelledTable = set_CancelledTable(C_dat)
    CustomerTable = pd.concat([rfmTable, CustomerTable, MeanBasketTable],
                           axis=1)
    
    CustomerTable = CustomerTable.reset_index()
    CancelledTable = CancelledTable.reset_index()
    # merge: keep customers with 0 cancellation or >0 cancellations
    ## discard customers with 1+ cancellation but 0 recorded invoice
    FinalTable = CustomerTable.merge(CancelledTable,on='CustomerID',how='left')
    FinalTable = FinalTable.fillna(0)
    
    FinalTable['C_NbArt'] = -FinalTable['C_NbArt']
    FinalTable['C_monetary'] = -FinalTable['C_monetary']
    FinalTable['net_frequency'] = FinalTable['frequency'] - FinalTable['C_NbInvoice']
    FinalTable['net_monetary'] = FinalTable['monetary_value'] - FinalTable['C_monetary']
    FinalTable['C_Art_ratio'] = FinalTable['C_NbArt']/FinalTable['NbArt']
    FinalTable['C_Inv_ratio'] = FinalTable['C_NbInvoice']/FinalTable['frequency']
    # checking(s)
    indx = []
    indx.extend(FinalTable[FinalTable['net_frequency']<0].index)
    indx.extend(FinalTable[FinalTable['net_monetary']<0].index)
    indx = np.unique(indx)
    FinalTable = FinalTable.drop(index=indx)
    
    FinalTable.index = FinalTable['CustomerID']
    FinalTable = FinalTable.drop(columns=['CustomerID'])
    
    return(FinalTable)