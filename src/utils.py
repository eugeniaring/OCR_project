import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def import_annot(path_annot,l_paths):
    ## add word length
    df_annot = pd.read_parquet(path_annot)
    df_annot['path'] = 'images/'+df_annot['image_id']+'.jpg'
    df_annot = df_annot[df_annot.path.isin(l_paths)]
    df_annot['word_length'] = df_annot['utf8_string'].str.len()
    return df_annot


def add_annotations(df,df_annot,l_paths):
    ## create dictionary
    diz_annot = {x:[] for x in l_paths}
    for el in diz_annot.keys():
        sample_df = df_annot[df_annot.path==el]
        if len(sample_df)!=0:
            for w_len in sample_df['word_length']:
                diz_annot[el].append(w_len)
    print(el)
    ## enrich data
    df['all_words_length'] = df['path'].map(lambda x: np.array(diz_annot[x]))
    df['present_text'] = df['all_words_length'].apply(lambda x: 1 if len(x)>0 else 0)
    df = df[df.present_text!=0]
    df['word_length'] = df['all_words_length'].apply(lambda x: x.max())
    df.drop(['all_words_length'],axis=1,inplace=True)
    df.reset_index(inplace=True,drop=True)
    return df

def create_split_col(df):
    train_idx, test_idx= train_test_split(df.index,test_size=0.75,random_state=123)
    df['split'] = 'test'
    df.loc[df.index.isin(train_idx),'split'] = 'train'  
    return df