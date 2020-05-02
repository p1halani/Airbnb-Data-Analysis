import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib

from . import dispatcher
from src.dataset import DataPreprocess
from . import config

FOLD_MAPPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == "__main__":
    df = pd.read_hdf(config.DATA_LISTINGS, 'seattle_listings')
    
    train_df = df[df.kfold.isin(FOLD_MAPPPING.get(config.FOLD))].reset_index(drop=True)
    valid_df = df[df.kfold==config.FOLD].reset_index(drop=True)

    ytrain = train_df.price.values
    yvalid = valid_df.price.values

    train_df = train_df.drop(["price", "kfold"], axis=1)
    valid_df = valid_df.drop(["price", "kfold"], axis=1)

    valid_df = valid_df[train_df.columns]

    label_encoders = {}
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        train_df.loc[:, c] = train_df.loc[:, c].astype(str).fillna("NONE")
        valid_df.loc[:, c] = valid_df.loc[:, c].astype(str).fillna("NONE")
        # df_test.loc[:, c] = df_test.loc[:, c].astype(str).fillna("NONE")
        lbl.fit(train_df[c].values.tolist() + 
                valid_df[c].values.tolist())
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders[c] = lbl

    # data is ready to train
    regr = dispatcher.MODELS[config.MODEL]
    regr.fit(train_df, ytrain)
    preds = regr.predict(valid_df)
    print(metrics.r2_score(yvalid, preds))

    joblib.dump(label_encoders, f"models/{config.MODEL}_{config.FOLD}_label_encoder.pkl")
    joblib.dump(regr, f"models/{config.MODEL}_{config.FOLD}.pkl")
    joblib.dump(train_df.columns, f"models/{config.MODEL}_{config.FOLD}_columns.pkl")
