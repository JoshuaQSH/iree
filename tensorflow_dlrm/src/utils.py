import time
import numpy as np
import functools
import pandas as pd

# Define a function that calculate the running time
def time_count(func):
    @functools.wraps(func)
    def runtime(*args, **kw):
        time_mean = []
        # Iteration time
        iter = 5
        for i in range(iter):
            start_time = time.time()
            func(*args, **kw)
            end_time = time.time()
        
        time_mean.append(end_time - start_time)

        print(f"Runtime: {np.mean(time_mean):.10f}")
  
    return runtime

class LabelEncoding():
    def __init__ (self):
        pass

    def fit(self, data, columns):

        self.columns = columns
        col_dic = {}
        inverse_dic = {}
        for col in columns:
            col_unique = data[col].unique()
            col_unique_dic = {}
            item_to_idx = {}
            for idx, item in enumerate(col_unique):
                col_unique_dic[item] = idx
                item_to_idx[idx] = item
            col_dic[col] = col_unique_dic
            inverse_dic[col] = item_to_idx

        self.item_to_idx = col_dic
        self.idx_to_item = inverse_dic

    def transform(self, data):
        df = data.copy()
        for col in self.columns:
            df[col] = df[col].map(self.item_to_idx[col])

        return df

    def inverse_transform(self, data):
        df = data.copy()
        for col in self.columns:
            df[col] = df[col].map(self.idx_to_item[col])

        return df

    def fit_transform(self, data, columns):
        df = data.copy()
        self.fit(df, columns)
        df = self.transform(df)

        return df

