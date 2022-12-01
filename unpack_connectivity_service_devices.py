
# IMPORTS
import pandas as pd
import pickle as pl
import os


def gzip_to_df(folder_path: str) -> pd.DataFrame:
    '''
    ## Unpacks all gzip files in folder (path) into pandas dataframe
    - all files are expected to be gzip
    '''

    print(f'unpacking -> {folder_path}')
    file_names = os.listdir(path=folder_path)
    dfs = []
    for i, file in enumerate(file_names):
        df_partial = pd.read_csv(f'{folder_path}{file}', compression='gzip', sep='|')
        dfs.append(df_partial)
    dfOrig = pd.concat(dfs).reset_index(drop=True)
    print(f'{len(dfs)} files added')
    
    try:
        assert len(dfOrig['SrcFileAbbr'].unique()) == 1
        print('df merge complete')
    except AssertionError:
        print('## WARNING! ##')
        print('tried integrating data from different source files')
        print(dfOrig['SrcFileAbbr'].unique())
    
    return dfOrig


def reformat_to_occ(df: pd.DataFrame) -> pd.DataFrame:
    '''
    ## Compiles unique FldVal occurrences for each FldKey in a batch normalized according to Stefan's template
    - Transforms each batch into a data point of shape (1, n), where n is number of FldKeys
    - Compiles samples into pandas DataFrame (dfOcc) of shape (m, n), where m is number of samples (batches)
    '''
    n_batches = len(df['BatchInstId'].unique())
    data = []

    for i, id in enumerate(df['BatchInstId'].unique()):
        ds = df.query(f'BatchInstId==@id')['FldKey']
        sample = ds.value_counts(sort=False)                # Counts of unique FldVals (only unique values in df)
        df_partial = pd.DataFrame(data=sample.values.reshape(1, -1), columns=sample.index)
        data.append(df_partial)
        print(f'batch {id}, {i+1} of {n_batches} added')
    dfOcc = pd.concat(data).reset_index(drop=True)

    for col in dfOcc.columns:
        dfOcc[col] = dfOcc[col].fillna(0).apply(pd.to_numeric).astype(int)

    return dfOcc


def main():
    abs_path = '/Users/oskarwallberg/Desktop/dataFiles/connectivity_service_devices_files/'
    dfOrig = gzip_to_df(folder_path=abs_path)
    print(dfOrig)
    print(dfOrig.info())

    # with open('PICKLED/connectivity_service_devices_original.pl', 'wb') as f:
    #     pl.dump(dfOrig, f)
    
    dfOcc = reformat_to_occ(df=dfOrig)
    print(dfOcc)
    print(dfOcc.info())

    # with open('PICKLED/connectivity_service_devices_occurrences.pl', 'wb') as f:
    #     pl.dump(dfOcc, f)
    


if __name__=='__main__':
    main()






