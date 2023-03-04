
# IMPORTS
import pandas as pd
import pickle as pl




def extract_amount(df_orig: pd.DataFrame, FldKey: str) -> pd.DataFrame:
    '''Docstring... '''
    df_FldKey = df_orig.query(f'FldKey==@FldKey')[['BatchInstId', 'FldVal', 'RowCnt']].reset_index(drop=True)
    return df_FldKey


def add_ordinal(df_FldKey: pd.DataFrame) -> pd.DataFrame:
    ''''''
    ordinal = []
    for i, batchId in enumerate(df_FldKey['BatchInstId'].unique().ravel()):        # batches are ordered according to Input file (chronological)
        ordinal += [i]*df_FldKey.query('BatchInstId==@batchId').shape[0]
    df_FldKey['Ordinal'] = ordinal
    return df_FldKey



def add_prevalence(df_FldKey: pd.DataFrame) -> pd.DataFrame:
    ''''''
    n_batches = df_FldKey.BatchInstId.nunique()
    df_prev = df_FldKey.groupby('FldVal').agg({'RowCnt': lambda x: len(list(x))/n_batches}).rename(columns={'RowCnt':'Prevalence'})
    transcription_dict = dict(zip(df_prev.index, df_prev.Prevalence))
    df_FldKey['Prevalence'] = df_FldKey['FldVal'].map(transcription_dict)
    return df_FldKey




def main():
    filepath = 'PICKLED/df_original_SAR.pl'
    with open(filepath, 'rb') as f:
        df_orig: pd.DataFrame = pl.load(f)

    # # potential interesting keys
    #     fldkeys = df_orig.FldKey.unique()
    #     for key in fldkeys:
    #         n = df_orig.query('FldKey==@key').FldVal.nunique()
    #         print(n, key)
    # quit()


    FldKey = 'Amount'

    df_FldKey = extract_amount(df_orig=df_orig, FldKey=FldKey)
    df_FldKey = add_ordinal(df_FldKey=df_FldKey)
    df_FldKey = add_prevalence(df_FldKey=df_FldKey)

    print(df_FldKey)
    print(df_FldKey.info())

    


    with open('PICKLED/df_SAR_FldKey.pl', 'wb') as f:
        pl.dump((df_FldKey, FldKey), f)




if __name__=='__main__':
    main()








