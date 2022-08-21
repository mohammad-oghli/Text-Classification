import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib


def extract_words(str, n):
    words = str.split()
    c = 0
    s_arr = []
    res = ""
    for w in words:
        res += w + " "
        c += 1
        if c % n == 0:
            res = res.strip()
            s_arr.append(res)
            res = ""
    return s_arr


def train_fx_model():
    df = pd.read_csv(r"BBC News Train 4 Categories.csv")
    df['category_id'] = df['Category'].factorize()[0]
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                            stop_words='english')
    features = tfidf.fit_transform(df.Text).toarray()
    labels = df.category_id
    model = LogisticRegression(random_state=0)
    x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index,
                                                                                     test_size=0.33, random_state=42,
                                                                                     shuffle=True)
    model.fit(x_train, y_train)
    #y_pred = model.predict(X_test)
    filename = "Completed_model.joblib"
    joblib.dump(model, filename)
