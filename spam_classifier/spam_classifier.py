import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer, TFBertModel


def text_preprocess(text):
    """ The function to remove punctuation,
    stopwords and apply stemming"""
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    porter = PorterStemmer()
    words = re.sub("[^a-zA-Z]", " ", text)
    words = [word.lower() for word in words.split() if word.lower()
             not in stop_words]
    words = [porter.stem(word) for word in words]
    return " ".join(words)


def build_model(bert_model):
    """ Creating model using BERT """
    input_word_ids = tf.keras.Input(shape=(64,), dtype='int32')
    attention_masks = tf.keras.Input(shape=(64,), dtype='int32')

    sequence_output = bert_model([input_word_ids, attention_masks])
    output = sequence_output[1]
    output = tf.keras.layers.Dense(32, activation='relu')(output)
    output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)

    model = tf.keras.models.Model(inputs=[input_word_ids, attention_masks], outputs=output)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    return model


class SpamClassifier:
    def __init__(self):
        self.device = tf.device("/GPU:0") if len(tf.config.list_physical_devices('GPU')) > 0 else tf.device("/device:CPU:0")
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        bert_model = TFBertModel.from_pretrained('bert-base-uncased')
        self.model = build_model(bert_model)
        self.model.load_weights("bert_model")

    def prepare(self, data):
        input_ids = []
        attention_masks = []

        for row in data:
            encoded = self.tokenizer.encode_plus(
                row,
                add_special_tokens=True,
                max_length=64,
                pad_to_max_length=True,
                return_attention_mask=True,
            )
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
        return np.array(input_ids), np.array(attention_masks)

    def predict(self, data: pd.DataFrame):
        data[data.columns[0]] = data[data.columns[0]].apply(text_preprocess)
        X, X_masks = self.prepare(data.values)
        predictions = self.model.predict_on_batch([X, X_masks])
        return [p[0] for p in predictions]
