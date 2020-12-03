import re
import numpy as np
import pandas as pd
import torch

import string
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from classifier import *
from sklearn.preprocessing import normalize
from sklearn import preprocessing
import sys
import argparse
from sklearn.metrics import confusion_matrix
from  utils import *
from sentence_transformers import SentenceTransformer, util

from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
import seaborn as sns
import matplotlib.pyplot as plt
import random
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix, classification_report
MAX_LENGTH=50



punct =[]
punct += list(string.punctuation)
punct += '’'
punct.remove("'")
def remove_punctuations(text):
    for punctuation in punct:
        text = text.replace(punctuation, ' ')
    return text

def preprocessing_emo(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    text = re.sub(r"http\S+", "", text)
    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    text=text.replace("..."," ")
#     text=text.replace("..","")
    text=text.replace(".","")
    text=text.replace("'s","")
    text=remove_punctuations(text)
    toks = nltk.tokenize.word_tokenize(text)
#     print(toks)
    text = [word for word in toks if word not in stopwords.words('english')]
    
    return text

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    text = re.sub(r"http\S+", "", text)
    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    text=text.replace("#","")

    return text


def clean_data(data):
    data = data[data.label != 'Neutral'] #remove neutral label for now
    data=data.drop_duplicates(subset='uid')
    
    data= data.groupby('label')
    data=data.apply(lambda x: x.sample(data.size().min()).reset_index(drop=True)) # balance data based on smaller class
    data['label_coded']=pd.Series(np.where(data.label.values == 'Favor', 0, 1),
          data.index)
    print(data.label.value_counts())
    return data

def norm(X):
    X=normalize(X, axis=0, norm='max')
    return X

def get_md_features(data):
    user_features= data[['uid','verified','acc_age','tff', 'rrt', 'recieved_likes','original_tweets','time_diff_sum','hashtag_num','dup_nums','url_num','dup_urlnums','alltweets_num']].copy()
    user_features['rlt'] = (user_features['recieved_likes']+1)/(user_features['original_tweets']+1)
    user_features['dplh']= (user_features['dup_nums']+1)/(user_features['alltweets_num']*(user_features['hashtag_num']+1))
    user_features['dplurl']= (user_features['dup_urlnums']+1)/(user_features['alltweets_num']*(user_features['url_num']+1))
    user_features['ratio_time_diffs'] = user_features['time_diff_sum'] / user_features['alltweets_num']
    user_features['rrt'] = user_features['rrt'].fillna(0)
    features=user_features.drop(columns=['uid','recieved_likes','original_tweets','time_diff_sum','hashtag_num','dup_nums','url_num','dup_urlnums','alltweets_num'])
    features["verified"] = features["verified"].astype(int)
    features=features.replace([np.inf, -np.inf], np.nan)
    features=features.fillna(0)
    feature_array=features.to_numpy()
    md_X=feature_array
    return norm(md_X)



def timeline_tweets_tocsv(path,data,user_data):
    timeline_tweet_data=pd.DataFrame([])
    ut_tweets=[]
    user_id=[]
    token_limit=10
    user_data['uid']=user_data['uid'].apply(str)
    user_data=user_data.set_index(['uid'])
    for user in data.uid.unique():
        for t_tweet in user_data.loc[[str(user)],'tweet'].values:
            if len(t_tweet.split())>=token_limit:
                ut_tweets.append(t_tweet)
                user_id.append(user)
    timeline_tweet_data['tweet']=ut_tweets
    timeline_tweet_data['label']=user_id
    timeline_tweet_data.to_csv(path+"/timeline_tweets.tsv",index=False)



# Create a function to tokenize a set of texts
def preprocessing_for_bert(data,max_len,tokenizer):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(str(sent)),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=max_len,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True,     # Return attention mask
            truncation=True
            )
        
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3


def read_data_from_file(file_path,test_size):
    train_df=pd.read_csv(file_path+"/train_main_sub.tsv")
    test_df=pd.read_csv(file_path+"/test_main_sub.tsv")
    subreddits_in_train=train_df.subreddit.unique()
    subreddits_in_test=test_df.subreddit.unique()
    subs_in_both=intersection(subreddits_in_train,subreddits_in_test)

    train_df=train_df.loc[train_df['subreddit'].isin(subs_in_both)]

    test_df=test_df.loc[test_df['subreddit'].isin(subs_in_both)]

    train_df = train_df[:20000]
    test_df=test_df[:4000]
    X_train=train_df['reddit'].values
    y_train=train_df['label'].values
    X_test=test_df['reddit'].values
    y_test=test_df['label'].values
    print(len(y_test))
    
    # # user_names=train_df[1].unique()
    # # user_names=np.unique((np.append(user_names,test_df[1].unique())))
    # # spliting test data into validation and test
    random_state = 50
    indices = np.arange(len(X_test))
    train_idx, val_idx, y_val, y_test= train_test_split(indices, y_test,stratify = y_test, test_size=test_size, random_state=random_state)
    X_val = X_test[train_idx]
    X_tst = X_test[val_idx]
    num_subs = 3146

    return X_train,y_train,X_val,y_val,X_tst,y_test,num_subs


def load_user_model(model_path,num_sub):
    #initialize model:
    model = SubRedditClassifier(num_sub)
    model.load_state_dict(torch.load(model_path+'full_best_model_state.bin'),strict=False)
    model.eval()
    # print(model)
    return model


def correct_data_format(file_path):
    records=[]
    # Using readlines() 
    file1 = open(file_path + 'test_main_sub.txt', 'r') 
    Lines = file1.readlines() 
    
    count = 0
    # Strips the newline character 
    for line in Lines:
        elements= line.split()
        reddit = " ".join(elements[3:])
        
        records.append([elements[1],elements[2],reddit])
    df = pd.DataFrame(records, columns=['label','subreddit','reddit']) 
    df.to_csv(file_path + 'test_main_sub.tsv',index=False)



def tokenize_input(X_train,X_val,X_test,load_dir):
    # tokenizer = RobertaTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # else:
    #     print("should be here")
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    # PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
    # tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    # train_inputs, train_masks = preprocessing_for_bert(X_train,MAX_LENGTH,tokenizer)
    # val_inputs, val_masks = preprocessing_for_bert(X_val,MAX_LENGTH,tokenizer)
    # test_inputs, test_masks = preprocessing_for_bert(X_test,MAX_LENGTH,tokenizer)
    # torch.save(train_inputs,f'{load_dir}/train_inputs.pt')
    # torch.save(train_masks,f'{load_dir}/train_masks.pt')
    # torch.save(val_inputs,f'{load_dir}/val_inputs.pt')
    # torch.save(val_masks,f'{load_dir}/val_masks.pt')
    # torch.save(test_inputs,f'{load_dir}/test_inputs.pt')
    # torch.save(test_masks,f'{load_dir}/test_masks.pt')

    train_inputs = torch.load(f'{load_dir}/train_inputs.pt')
    train_masks=torch.load(f'{load_dir}/train_masks.pt')
    val_inputs = torch.load(f'{load_dir}/val_inputs.pt')
    val_masks = torch.load(f'{load_dir}/val_masks.pt')
    test_inputs = torch.load(f'{load_dir}/test_inputs.pt')
    test_masks = torch.load(f'{load_dir}/test_masks.pt')
    train_inputs = train_inputs[:20000][:]
    train_masks = train_masks[:20000][:]
    val_inputs = val_inputs[:2000][:]
    val_masks = val_masks[:2000][:]
    test_inputs = test_inputs[:2000][:]
    test_masks = test_masks[:2000][:]
    print(train_inputs.size())
 
    return train_inputs, train_masks,val_inputs,val_masks,test_inputs, test_masks