import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader,Subset


class JigsawDataset(Dataset):
    '''
    Dataset object toxic Jigsaw wikipedia comments

    args:
        input_file (str): data file with both observations and labels
        whitelist (list[str]): columns to keep from dataframe
    '''
    
    def __init__(self, input_file, whitelist=['toxic','severe_toxic','obscene','threat','insult','identity_hate']):
        # essentially we want output to be: ("string with comment text",[one hot vector of target labels])
        df = pd.read_csv(input_file)
        self.comments = df['comment_text'].values
        self.labels = df[whitelist].values
    
    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        return (self.comments[idx], self.labels[idx])
    
class YoutubeDataset(Dataset):
    def __init__(self, input_file, whitelist=['IsToxic','IsObscene','IsHatespeech','IsThreat']):
        mappings = {'IsToxic':'toxic',
                    'IsObscene':'obscene',
                    'IsHatespeech':'identity_hate',
                    'IsThreat':'threat'}
        
        # essentially we want output to be: ("string with comment text",[one hot vector of target labels])
        df = pd.read_csv(input_file)
        self.comments = df['Text'].values
        df[whitelist] = df[whitelist].astype(int)
        self.labels = df[whitelist].values
    
    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        return (self.comments[idx], self.labels[idx])


def model_output_to_dict(prediction, whitelist=['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult','identity_attack']):
    '''
    Takes pipeline output from form [[{},{},{}..]] to {} 
    '''
    result_dict = {}
    prediction = prediction[0]
    for i in range(len(prediction)):
        pair = list(prediction[i].values())
        if pair[0] in whitelist:
            result_dict[pair[0]] = pair[1]

    return result_dict


