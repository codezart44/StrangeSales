
# IMPORTS
import pandas as pd
import pickle as pl
import numpy as np
import os


def unpack_gzip(folder_path: str) -> list:
    '''
    ## Unpacks all gzip files in folder (path) into pandas dataframe
    - all files are expected to be gzip
    - divides batches into:
    '''

    print(f'unpacking -> {folder_path}')
    file_names = list(filter(lambda file_name: '.txt.gz' in file_name, os.listdir(path=folder_path)))
    all_batches = np.array([pd.read_csv(f'{folder_path}{file}', compression='gzip', sep='|') for file in file_names], dtype=pd.DataFrame)
    df_orig = pd.concat(all_batches).reset_index(drop=True)
    print(df_orig['BatchInstId'].nunique())

    try:
        assert len(df_orig['SrcFileAbbr'].unique()) == 1
        print('df merge complete')
    except AssertionError:
        print('## WARNING! ##')
        print('tried merging data from different source files')
        print(df_orig['SrcFileAbbr'].unique())
    
    return df_orig





def main():
    abs_path = '/Users/oskarwallberg/Desktop/dataFiles/connectivity_service_devices_files/'
    df_orig: pd.DataFrame = unpack_gzip(folder_path=abs_path)
    print(df_orig)
    print(df_orig.info())

    # with open('PICKLED/df_original_CSD.pl', 'wb') as f:
    #     pl.dump(df_orig, f)
    




if __name__=='__main__':
    main()




