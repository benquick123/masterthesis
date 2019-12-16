import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical


def load_data(encode=True):
    X1_train = pickle.load(open("/opt/workspace/host_storage_hdd/cotrain_data/preprocessed-data/X1_train_init.p", "rb"))
    X2_train = pickle.load(open("/opt/workspace/host_storage_hdd/cotrain_data/preprocessed-data/X2_train_init.p", "rb"))
    y_train = pickle.load(open("/opt/workspace/host_storage_hdd/cotrain_data/preprocessed-data/y1_train_init.p", "rb"))
    
    X1_unlabel = pickle.load(open("/opt/workspace/host_storage_hdd/cotrain_data/preprocessed-data/X1_unlabeled.p", "rb"))
    X2_unlabel = pickle.load(open("/opt/workspace/host_storage_hdd/cotrain_data/preprocessed-data/X2_unlabeled.p", "rb"))
    
    X1_test = pickle.load(open("/opt/workspace/host_storage_hdd/cotrain_data/preprocessed-data/X1_test.p", "rb"))
    X2_test = pickle.load(open("/opt/workspace/host_storage_hdd/cotrain_data/preprocessed-data/X2_test.p", "rb"))
    y_test = pickle.load(open("/opt/workspace/host_storage_hdd/cotrain_data/preprocessed-data/y1_test.p", "rb"))
    
    if encode:
        labels = np.array(np.unique(y_train), dtype="int")
        _y_train, _y_test = np.zeros((y_train.shape[0], len(labels))), np.zeros((y_test.shape[0], len(labels)))
        for label in labels:
            _y_train[y_train == label, label] = 1
            _y_test[y_test == label, label] = 1
        
        y_train = _y_train
        y_test = _y_test
  
    return X1_train, X2_train, y_train, X1_unlabel, X2_unlabel, X1_test, X2_test, y_test


def load_data_v2():
    df = pd.read_csv("/opt/workspace/host_storage_hdd/cotrain_data/GrammarandProductReviews.csv")
    df['target'] = df['reviews.rating'] <= 4

    df['reviews.text'] = df['reviews.text'].astype('str')
    
    train_text, test_text, train_y, test_y = train_test_split(df['reviews.text'], df['target'],test_size = 0.3)
    train_text, unsup_text, train_y, _ = train_test_split(train_text, train_y, test_size=0.98)
    test_text, val_text, test_y, val_y = train_test_split(test_text, test_y, test_size=0.1)
    
    MAX_NB_WORDS = 20000

    # get the raw text data
    texts_train = train_text.astype(str)
    texts_test = test_text.astype(str)
    texts_val = val_text.astype(str)
    texts_unsup = unsup_text.astype(str)

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, char_level=False)
    tokenizer.fit_on_texts(texts_train)
    sequences = tokenizer.texts_to_sequences(texts_train)
    sequences_val = tokenizer.texts_to_sequences(texts_val)
    sequences_test = tokenizer.texts_to_sequences(texts_test)
    sequences_unsup = tokenizer.texts_to_sequences(texts_unsup)

    word_index = tokenizer.word_index
    
    index_to_word = dict((i, w) for w, i in tokenizer.word_index.items())
    
    X1_sequences = [el[:len(el) // 2] for el in sequences]
    X2_sequences = [el[len(el) // 2:] for el in sequences]
    X1_sequences_test = [el[:len(el) // 2] for el in sequences_test]
    X2_sequences_test = [el[len(el) // 2:] for el in sequences_test]
    X1_sequences_val = [el[:len(el) // 2] for el in sequences_val]
    X2_sequences_val = [el[len(el) // 2:] for el in sequences_val]

    X1_sequences_unsup = [el[:len(el) // 2] for el in sequences_unsup]
    X2_sequences_unsup = [el[len(el) // 2:] for el in sequences_unsup]
    
    MAX_SEQUENCE_LENGTH = 100

    # pad sequences with 0s
    x1_train = pad_sequences(X1_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    x1_val = pad_sequences(X1_sequences_val, maxlen=MAX_SEQUENCE_LENGTH)
    x1_test = pad_sequences(X1_sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
    x1_unsup = pad_sequences(X1_sequences_unsup, maxlen=MAX_SEQUENCE_LENGTH)
    x2_train = pad_sequences(X2_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    x2_val = pad_sequences(X2_sequences_val, maxlen=MAX_SEQUENCE_LENGTH)
    x2_test = pad_sequences(X2_sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
    x2_unsup = pad_sequences(X2_sequences_unsup, maxlen=MAX_SEQUENCE_LENGTH)
    
    y_train = train_y
    y_val = val_y
    y_test = test_y

    y_train = to_categorical(np.asarray(y_train))
    y_val = to_categorical(np.asarray(y_val))
    y_test = to_categorical(np.asarray(y_test))
    
    return x1_train, x2_train, y_train, x1_unsup, x2_unsup, x1_val, x2_val, y_val
