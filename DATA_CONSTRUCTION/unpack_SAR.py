
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
    df_orig = pd.concat(df_chunks)

    try:
        assert df_orig['SrcFileAbbr'].nunique() == 1
        print('df merge complete')
    except AssertionError:
        print('## WARNING! ##')
        print('tried integrating data from different source files')
        print(df_orig['SrcFileAbbr'].unique())

    return df_orig



def main():
    abs_path = '/Users/oskarwallberg/Desktop/dataFiles/SpiderAccountRows_files/Input.csv'
    df_orig = csv_to_df(file_path=abs_path)
    print(df_orig)
    print(df_orig.info())

    # with open('PICKLED/df_original_SAR.pl', 'wb') as f:
    #     pl.dump(df_orig, f)
    



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


