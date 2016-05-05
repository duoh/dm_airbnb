import numpy as np
import pandas as pd

def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k=5, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def score_predictions(preds, truth, n_modes=5):
    assert(len(preds)==len(truth))
    r = pd.DataFrame(0, index=preds.index, columns=preds.columns, dtype=np.float64)
    for col in preds.columns:
        r[col] = (preds[col] == truth) * 1.0
    score = pd.Series(r.apply(ndcg_at_k, axis=1, reduce=True), name='score')
    return score


import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer

def dcg_score(y_true, y_score, k=5):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gain = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


def ndcg_score(ground_truth, predictions, k=5):
    lb = LabelBinarizer()
    lb.fit(range(len(predictions) + 1))
    T = lb.transform(ground_truth)
    scores = []
    for y_true, y_score in zip(T, predictions):
        actual = dcg_score(y_true, y_score, k)
        best = dcg_score(y_true, y_true, k)
        score = float(actual) / float(best)
        scores.append(score)
    return np.mean(scores)


ndcg_scorer = make_scorer(ndcg_score, needs_proba=True, k=5)

import numpy as np
import pandas as pd
import xgboost as xb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
import operator

np.random.seed(0)

data = pd.read_csv('/Users/duoh/Documents/Mahidol/Data Mining/project/preprocessData.csv')

#Sampling
#np.random.seed(42)
#samples = np.random.choice(piv_train, 10000)
#X_train = vals[samples]
#y_train = le.fit_transform(labels)[samples]

labels = data['country_destination'].values
data = data.drop(['country_destination'], axis=1)
le = LabelEncoder()
train_num = round(data.shape[0]*0.7)
vals = data.values
trains = vals[:train_num]
pred = le.fit_transform(labels[:train_num])
trains_test = vals[train_num:]
labels_test = labels[train_num:]

#pred = le.fit_transform(data['country_destination'].values)
#data = data.drop(['country_destination'], axis=1)
#trains = data.values

model = KNeighborsClassifier()

model = LogisticRegression()

model = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)

#kf = KFold(len(X), n_folds=10, random_state=42)
#score = cross_val_score(model, trains, pred, scoring=ndcg_scorer)

#fearture importance
params = {'learning_rate':0.3,
        'max_depth': 6,
        'n_estimators':25,
        'objective': 'multi:softprob',
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'num_class': 12,
        'seed': 0}
dtrain = xgb.DMatrix(trains, pred)
trained_model = xgb.train(params=params, dtrain=dtrain)
import_feature = trained_model.get_fscore()
import_df = pd.DataFrame(sorted(import_feature.items(), key=operator.itemgetter(1)),columns=['feature','fscore'])
import_df.iloc[-20:,:].plot(x='feature',y='fscore',kind='barh')
#plt.show()

model.fit(trains, pred)
pred_test = model.predict_proba(trains_test)

cts = []
size = len(trains_test)
for i in range(size):
    cts.append(le.inverse_transform(np.argsort(pred_test[i])[::-1])[:5].tolist())

df = pd.DataFrame(cts,columns=[0,1,2,3,4])
scores = score_predictions(df,labels_test)
score = float(sum(scores))/float(len(scores))
round(score,6)