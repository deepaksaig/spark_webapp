from flask import Flask, render_template, request, url_for
import flask
import re
import pickle
import numpy as np
import pandas as pd
import os
import sys
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dropout, Conv1D, MaxPool1D, GRU, LSTM, Dense

import matplotlib.pyplot as plt




sys.path.append("C:\spark-2.3.2-bin-hadoop2.7\python")
sys.path.append("C:\spark-2.3.2-bin-hadoop2.7\python\lib\py4j-0.10.7-src.zip")
from pyspark.sql import SparkSession









# Initialise the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'people_photo')

@app.route('/image', methods=['GET', 'POST'])
def lionel():
    return render_template('index.html')

# Set up the main route

@app.route('/',methods=['GET', 'POST'])
def review():
    spark = SparkSession \
        .builder \
        .appName(" movie data") \
        .getOrCreate()
    data = spark.read \
        .format("csv") \
        .option("Header", "true") \
        .load("static/finaldf.csv")
    data2 = data.drop('fx', 'sno', 'id')
    deepak = 'null'
    def moviereview(name):
        movie_details = data2.filter(data2.title == name)
        rating = [x[1] for x in movie_details.toLocalIterator()]
        sentiment = [x[2] for x in movie_details.toLocalIterator()]
        imdb_rating = float(rating[0])
        sentiment1 = float(sentiment[0])
        rating_per = (imdb_rating * 10)
        rating_agr = (100 - rating_per)
        print(imdb_rating)
        print(rating_per)
        print(rating_agr)
        print(sentiment)
        s1= str(rating_per)
        s2 = str(rating_agr)
        if sentiment1 == 1:
            example = ("THE REVIEW ANALYSIS OF THE MOVIE CAME OUT TO BE POSITIVE")
        else:
            example = ("THE REVIEW ANALYSIS OF THE MOVIE CAME OUT TO BE NEGATIVE")

        example1 = str(s1 + " CAME OUT TO BE POSITIVE")
        example2 = str(s2 +  " CAME OUT TO BE NEGATIVE")
        labels = 'Positive', 'Negative'
        sizes = [rating_per, rating_agr]
        explode = (0.2, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=0)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.savefig('static/download.png')
        list = [example,example1,example2]
        return list

    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return (flask.render_template('input.html'))

    if flask.request.method == 'POST':
        # Extract the input
        name = flask.request.form['name']
        moviereview(name)
        list1 = moviereview(name)
        message = list1[0]
        message2 = list1[1]
        message3 = list1[2]

    return flask.render_template('index.html',Result = message,Result2 = message2,Result3 = message3,name = name )





@app.route('/main', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        Review = flask.request.form['Review']

        # Make DataFrame for model
        input_variables = pd.DataFrame([[Review]],
                                       columns=['Review'],
                                       dtype=str,
                                       index=['input'])

        outputName = "test" + ".csv"
        input_variables.to_csv(outputName, index=False, quoting=3, )

        def reviewWords(review, method):
            data_train_Exclude_tags = re.sub(r'<[^<>]+>', " ", review)  # Excluding the html tags
            data_train_num = re.sub(r'[0-9]+', 'number', data_train_Exclude_tags)  # Converting numbers to "NUMBER"
            data_train_lower = data_train_num.lower()  # Converting to lower case.
            data_train_no_punctuation = re.sub(r"[^a-zA-Z]", " ", data_train_lower)

            # ussing stop words.
            # After using stop words, training accuracy increases, but testing accuracy decreases in Kaggle.
            # This method might overfit the training data.
            if method == "Stop Words":
                # print("Processing dataset with stop words...")
                data_train_split = data_train_no_punctuation.split()  # Splitting into individual words.
                stopWords = set(stopwords.words("english"))
                meaningful_words = [w for w in data_train_split if not w in stopWords]  # Removing stop words.
                return (" ".join(meaningful_words))

            if method == "Nothing":
                # print("Processing dataset without porter stemming and stop words...")
                return data_train_no_punctuation

        def training_Validation_Data(cleanWords, data_train):

            X = cleanWords
            y = data_train["sentiment"]

            test_start_index = int(data_train.shape[0] * .8)

            x_train = X[0:test_start_index]
            y_train = y[0:test_start_index]
            x_val = X[test_start_index:]
            y_val = y[test_start_index:]

            return x_train, y_train, x_val, y_val

        # Reading the Data
        data_train = pd.read_csv("F:/Sentiment-Analysis-IMDb-Movie-Review/labeledTrainData.tsv", delimiter="\t")
        data_test = pd.read_csv("test.csv", delimiter="\t")

        # Input the value, whether you want to include porter stemming, stopwords.


        method = "Nothing"

        # Input the value, whether you want to run the model on LSTM RNN or GRU RNN.


        lstm = True

        # Let's process all the reviews together of train data.


        print("---Review Processing Done!---\n")

        # Splitting the data into tran and validation
        cleanWords = []
        for i in range(data_train['review'].size):
            cleanWords.append(reviewWords(data_train["review"][i], method))
        print("---Review Processing Done!---\n")

        # Splitting the data into tran and validation
        x_train, y_train, x_val, y_val = training_Validation_Data(cleanWords, data_train)

        # There is a data leakage in test set.


        # Processing text dataset reviews.
        testcleanWords = []
        for i in range(data_test['Review'].size):
            testcleanWords.append(reviewWords(data_test["Review"][i], method))
        print("---Test Review Processing Done!---\n")

        # Generate the text sequence for RNN model
        np.random.seed(1000)
        num_most_freq_words_to_include = 5000
        MAX_REVIEW_LENGTH_FOR_KERAS_RNN = 500  # Input for keras.
        embedding_vector_length = 32
        length = 1

        all_review_list = x_train + x_val

        tokenizer = Tokenizer(num_words=num_most_freq_words_to_include)
        tokenizer.fit_on_texts(all_review_list)

        # tokenisingtrain data
        train_reviews_tokenized = tokenizer.texts_to_sequences(x_train)
        x_train = pad_sequences(train_reviews_tokenized, maxlen=MAX_REVIEW_LENGTH_FOR_KERAS_RNN)  # 20,000 x 500

        # tokenising validation data
        val_review_tokenized = tokenizer.texts_to_sequences(x_val)
        x_val = pad_sequences(val_review_tokenized, maxlen=MAX_REVIEW_LENGTH_FOR_KERAS_RNN)  # 5000 X 500

        test_review_tokenized = tokenizer.texts_to_sequences(testcleanWords)
        x_test = pad_sequences(test_review_tokenized, maxlen=length)  # 5000 X 500

        def RNNModel(lstm):
            model = Sequential()
            model.add(Embedding(input_dim=num_most_freq_words_to_include,
                                output_dim=embedding_vector_length,
                                input_length=1))

            model.add(Dropout(0.2))
            model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
            model.add(LSTM(100))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            return model

        themodel = RNNModel(lstm)
        weights_path = "F:\spark-webapp\model.h5"
        themodel.load_weights(weights_path)
        themodel.summary()

        ytest_prediction = themodel.predict(x_test)

        ytest_prediction = np.array(ytest_prediction).reshape((1,))


        # Copy the predicted values to pandas dataframe with an id column, and a sentiment column.
        output = pd.DataFrame(data={"sentiment": ytest_prediction})
        outputName = "prediction" + ".csv"
        output.to_csv(outputName, index=False, quoting=3, )

        # Get the model's prediction
        predictionvalue = pd.read_csv("prediction.csv", delimiter="\t")
        prediction = predictionvalue['sentiment'].iloc[0]
        print(prediction)
        if prediction > 0.50:
            value = "pos"
        elif prediction < 0.49:
            value = "neg"
        else:
            value = "neutral"

        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',
                                     original_input={'Review':Review},
                                     result=value,
                        )

if __name__ == '__main__':
    app.run()