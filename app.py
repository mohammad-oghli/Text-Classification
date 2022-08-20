import streamlit as st
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from train_model import train_fx_model


def classify_text(sentence):
    '''
    Classify text to different categories using trained ML model
    :param
    sentence(str): Input text

    :return
    prediction(dict): Dictionary containing the category of classification
    '''
    # Train the model if pretrained Completed_model.joblib file is missing
    #train_fx_model()
    df = pd.read_csv(r"BBC News Train 4 Categories.csv")
    df['category_id'] = df['Category'].factorize()[0]
    category_id_df = df[['Category', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'Category']].values)
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                            stop_words='english')
    features = tfidf.fit_transform(df.Text).toarray()
    filename = "Completed_model.joblib"
    loaded_model = joblib.load(filename)
    # category to id
    # {'business': 0, 'tech': 1, 'politics': 2, 'entertainment': 3}
    if sentence:
        sent_array = [sentence]
        text_features = tfidf.transform(sent_array)
        predictions = loaded_model.predict(text_features)
        # for word, predicted in zip(sent_array, predictions):
        #     print(f"{word} :{id_to_category[predicted]}")
        return {predictions[0]: id_to_category[predictions[0]]}
    return "No valid category"


def st_ui():
    '''
    Render the User Interface of the application endpoints
    '''
    st.title("Text Classification")
    st.caption("Text Features Extraction")
    st.info("Developed by Oghli")
    st.header("Enter text to classify it according to the following categories:")
    st.write("###### • Business")
    st.markdown("###### • Technology")
    st.write("###### • Politics")
    st.write("###### • Entertainment")
    st.markdown("""---""")
    in_text = st.text_area(label='Input Text', placeholder='type your text')
    cls_btn = st.button('Classify')
    if cls_btn:
        classify_result = classify_text(in_text)
        if type(classify_result) is dict:
            st.markdown("""---""")
            st.subheader("Classification Result")
            st.success(f"#### Category: {list(classify_result.values())[0].capitalize()}")
        else:
            st.error(classify_result)


if __name__ == "__main__":
    # render the app using streamlit ui function
    st_ui()
#     sentence = '''
# Buying and selling stocks and shares has always involved a lot of third parties, such as brokers and the stock exchange itself.
# Here is how trading works:
# The buyer or seller initiates the trade.
# A broker sends a transaction to a stock exchange.
# The transaction is matched with another party (the counterparty).
# The transaction is sent to Central Counterparty Clearing House for risks evaluation.
# '''
#     print(classify_text(sentence))
