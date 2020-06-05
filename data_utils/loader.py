import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch


class Loader():
    def __init__(self, data_path, keep_actors=None, min_revenue=1e+6, min_movies=20, seed=101):
        self.min_revenue = min_revenue
        self.min_movies = min_movies
        if keep_actors is None:
            keep_actors = []
            
        data_actors = pd.read_csv(data_path, index_col=0)
        cols_20 = ['title_x', 'revenue']
        for col in data_actors.columns[2:-1]:
            if col in keep_actors: 
                continue
            elif np.sum(data_actors[col]) >= min_movies:
                cols_20.append(col)

        data_rev = data_actors[cols_20 + keep_actors]
        
        mask = (data_rev['revenue'] > min_revenue)
        for actor in keep_actors:
            mask |= (data_rev[actor] == 1)
        
        self.data_X = data_rev.loc[mask, data_rev.columns.difference(['revenue'])]
        self.data_y = data_rev.loc[mask, 'revenue']

        self.X = self.data_X.drop("title_x", axis=1).to_numpy(dtype='float32')
        self.y = np.log1p(self.data_y.to_numpy(dtype='float32'))
        
        self.X_train, self.X_test, self.y_train, self.y_test, self.data_train, self.data_test = train_test_split(self.X, self.y, self.data_X, test_size=0.2, random_state=seed)
        
    def get_train_test(self):
        return torch.tensor(self.X_train), torch.tensor(self.X_test), \
               torch.tensor(self.y_train), torch.tensor(self.y_test)
    
    def get_by_mask(self, train_mask, test_mask):
        X_train = self.data_train.loc[train_mask, :].to_numpy(dtype='float32')
        y_train = self.y_train[train_mask]
        
        X_test = self.data_test.loc[test_mask, :].to_numpy(dtype='float32')
        y_test = self.y_test[test_mask]
        
        return torch.tensor(X_train), torch.tensor(X_test), \
               torch.tensor(y_train), torch.tensor(y_test)
    
    def mask_actors(self, train_mask, test_mask):
        X_train = self.data_train.to_numpy(dtype='float32')
        X_train[train_mask] = 0
        
        X_test = self.data_test.to_numpy(dtype='float32')
        X_test[test_mask] = 0
        
        return torch.tensor(X_train), torch.tensor(X_test), \
               torch.tensor(self.y_train), torch.tensor(self.y_test)
    
    