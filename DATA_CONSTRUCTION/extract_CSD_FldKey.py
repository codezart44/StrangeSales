

# IMPORTS
import pandas as pd
import pickle as pl




def extract_FldKey(df_orig: pd.DataFrame, FldKey: str) -> pd.DataFrame:
    ''''''
    # df_amount.FldVal = df_amount.FldVal.str.replace(',', '.').astype(float)

    df_FldKey = df_orig.query('FldKey==@FldKey')[['BatchInstId', 'FldVal', 'Count']].reset_index(drop=True)
    return df_FldKey



def add_ordinal(df_FldKey: pd.DataFrame) -> pd.DataFrame:
    ''''''
    ordinal = []
    for i, batchId in enumerate(df_FldKey.BatchInstId.unique()):
        ordinal += [i]*df_FldKey.query('BatchInstId==@batchId').shape[0]
    df_FldKey['Ordinal'] = ordinal
    return df_FldKey



def add_prevalence(df_FldKey: pd.DataFrame) -> pd.DataFrame:
    '''
    df_prev has FldVal categories as index (from groupby), thus can through 
    transcription_dict map each prevalence to the corresponding FldVal
    '''
    n_batches = df_FldKey.BatchInstId.nunique()
    df_prev = df_FldKey.groupby('FldVal').agg({'Count': lambda x: len(list(x))/n_batches}).rename(columns={'Count':'Prevalence'})
    transcription_dict = dict(zip(df_prev.index, df_prev.Prevalence))
    df_FldKey['Prevalence'] = df_FldKey['FldVal'].map(transcription_dict)
    return df_FldKey




def main():
    with open('PICKLED/df_original_CSD.pl', 'rb') as f:
        df_orig: pd.DataFrame = pl.load(f)


    # potential interesting keys
    fldkeys = df_orig.FldKey.unique()
    for key in fldkeys:
        n = df_orig.query('FldKey==@key').FldVal.nunique()
        print(n, key)

    FldKey = 'modelName'             # <-- NOTE!
    
    
    df_FldKey = extract_FldKey(df_orig=df_orig, FldKey=FldKey)
    df_FldKey = add_ordinal(df_FldKey=df_FldKey)
    df_FldKey = add_prevalence(df_FldKey=df_FldKey)


    print(df_FldKey)
    print(df_FldKey.info())



    with open('PICKLED/df_CSD_FldKey.pl', 'wb') as f:
        pl.dump((df_FldKey, FldKey), f)




        
        






if __name__=='__main__':
    main()