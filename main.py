import pandas as pd
from fastapi import FastAPI
from joblib import load
import numpy as np

app = FastAPI()


@app.get("/api/recsys")
async def get_best_events(user_id: int):
    nbrs = load('model-registry/item2item_model.joblib')
    sparse_matrix = pd.read_csv('data/sparse_matrix.csv', index_col=0)
    _, indices = nbrs.kneighbors(sparse_matrix.T)

    user_items = sparse_matrix.T.loc[sparse_matrix.loc[user_id][sparse_matrix.loc[user_id] == 1].index]

    _, indices = nbrs.kneighbors(user_items)
    indices, counts = np.unique(indices.flatten(), return_counts=True)
    raw_predicts = list(list(zip(*sorted(list(zip(indices.tolist(), counts.tolist())), key=lambda el: -el[1])))[0])
    already_was = set(sparse_matrix.T.iloc[raw_predicts].index) & set(user_items.index)
    best_events = sparse_matrix.T.iloc[raw_predicts].drop(index=already_was)
    return {"events": list(best_events.index.astype(int))}