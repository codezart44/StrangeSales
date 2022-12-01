
# IMPORTS
import pandas as pd
import pickle as pl


def csv_to_df(file_path: str) -> pd.DataFrame:
    '''
    ## Reads path to csv and inserts data into pandas DataFrame
    - data is assumed to follow Stefan's template for normalized data
    - used to read in large csv files
    - csv must only contain data from one source file, i.e. SrcFileAbbr must be same for all data

    '''
    print(f'unpacking -> {file_path}')
    df_chunks = pd.read_csv(file_path, chunksize=10_000, low_memory=False)
    dfOrig = pd.concat([df for df in df_chunks])

    try:
        assert dfOrig['SrcFileAbbr'].nunique() == 1
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
    abs_path = '/Users/oskarwallberg/Desktop/dataFiles/SpiderAccountRows_files/Input.csv'
    dfOrig = csv_to_df(file_path=abs_path)
    print(dfOrig)
    print(dfOrig.info())

    # with open('PICKLED/SpiderAccountRows_original', 'wb') as f:
    #     pl.dump(dfOrig, f)

    dfOcc = reformat_to_occ(df=dfOrig)
    print(dfOcc)
    print(dfOcc.info())

    # with open('PICKLED/SpiderAccountRows_occurrences', 'wb') as f:
    #     pl.dump(dfOcc, f)



if __name__=='__main__':
    main()









'''
# FULL CYCLE
1. Fetch raw good (/bad if given) data, given by Stefan
2. Insert into dataframe, extract relevant data
3. Pickle data

4. ML model pipline (class)
5. Train model, with optimal params, test...
6. Pickle model
7. Reuse on unclassified data
8. Distribute data to trainginset (accumulates over time)
'''


