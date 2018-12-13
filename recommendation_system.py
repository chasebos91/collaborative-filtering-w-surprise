import surprise as sp
from surprise import Reader
from surprise import accuracy
import pandas as pd
from collections import defaultdict


from surprise import Dataset

# data preparation
def data_preparation():
    reader = Reader(line_format='user item rating', sep=',', skip_lines=1)

    d = pd.read_csv("~/Desktop/Tufts/Fall2018/COMP135/Project3/trainset.csv")
    trainset = Dataset.load_from_df(d[["user", "item", "rating"]], reader=reader)

    d = pd.read_csv("~/Desktop/Tufts/Fall2018/COMP135/Project3/testset.csv")
    testset = Dataset.load_from_df(d[["user", "item", "rating"]], reader=reader)

    d = pd.read_csv("~/Desktop/Tufts/Fall2018/COMP135/Project3/testset.csv", names=['user', 'item', 'rating'], header=1)
    tset = defaultdict(list)

    for row in d.iterrows():
        tset[str(row[1][0])].append((row[1][1], row[1][2]))

    return trainset, testset, tset

def model_select(trainset, k):
    param_grid = {'n_epochs': [1, 5, 10], 'lr_all': [0.01 , .1, .05],
                  'reg_all': [.01, .5, .8]}

    model = sp.model_selection.GridSearchCV(sp.SVD, param_grid, measures=['mae'], cv=k)

    model.fit(trainset)

    #print(model.cv_results)

    print(model.best_score['mae'])

    print(model.best_params['mae'])

    return model

def test_model(model):

    reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
    fold_files = [('~/Desktop/Tufts/Fall2018/COMP135/Project3/trainset.csv',
                   '~/Desktop/Tufts/Fall2018/COMP135/Project3/testset.csv')]

    pdkfold = sp.model_selection.split.PredefinedKFold()
    clf = model.best_estimator['mae']
    data = Dataset.load_from_folds(fold_files, reader=reader)

    for train, test in pdkfold.split(data):

        clf.fit(train)
        preds = clf.test(test)
        accuracy.mae(preds)

def k_recommend(model, k, testset):

    reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
    fold_files = [('~/Desktop/Tufts/Fall2018/COMP135/Project3/trainset.csv',
                   '~/Desktop/Tufts/Fall2018/COMP135/Project3/testset.csv')]

    pdkfold = sp.model_selection.split.PredefinedKFold()
    clf = model.best_estimator['mae']
    data = Dataset.load_from_folds(fold_files, reader=reader)

    for train, test in pdkfold.split(data):
        clf.fit(train)
        test1 = train.build_anti_testset()
        preds = clf.test(test1)

    top_n = defaultdict(list)

    for uid, iid, true_r, est, _ in preds:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:k]
    """
    for uid, user_ratings in top_n.items():
        print(uid, [iid for (iid, _) in user_ratings])
    for uid, user_ratings in top_n.items():
        print uid, user_ratings
    """

    for uid in top_n:
        i = 0
        for iid in top_n[uid]:
            found = False
            for iid2 in testset[uid]:
                if iid[0] == str(iid2[0]):
                    a = iid[0]
                    top_n[uid].remove(top_n[uid][i])
                    top_n[uid].insert(i,(a, iid2[1]))
                    found = True
                    i += 1
                    break
            if found == False:
                a = iid[0]
                top_n[uid].remove(top_n[uid][i])
                top_n[uid].insert(i,(a, 2))
                i += 1

    total_sum = 0.0
    user_sum = 0.0
    us_rec = []
    for uid in top_n:
        i = 0.0
        for iid in top_n[uid]:
            i += 1.0
            user_sum += iid[1]
        total_sum += float(user_sum / i)
        us_rec.append(user_sum / i)
        user_sum = 0.0

    #print us_rec
    print "Average rating: ", (total_sum/float(len(top_n)))

"""
#driver - uncomment to run

trainset, testset, tset = data_preparation()

best = model_select(trainset, 5)
test_model(best)

k_recommend(best, 5, tset)
"""

