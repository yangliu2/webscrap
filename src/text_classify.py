'''
' ideas based on https://arxiv.org/abs/1408.5882 and a lot of searching the internet
' The code requires tensorflow 1.0 and keras 1.2.2
'''
from __future__ import print_function


import os
import numpy as np
import pickle
import sys
import argparse
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.models import model_from_json

from random import randint

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

import sys

FLAGS = None

MAX_POOL_SIZE = 5
CNN_SIZE = 128 # or 128 seem to best performing in use cases tested
LSTM_SIZE = 128

def main():
    global FLAGS
    setup_tf_session()
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--text_dir',
      type=str,
      default='../data',
      help='Path to folders of labeled text files.'
    )
    parser.add_argument(
      '--glove_dir',
      type=str,
      default='../../glove',
      help='Directory where to find glove trained models.'
    )
    parser.add_argument(
      '--mode',
      type=str,
      default='train',
      help='if we are training or testing.'
    )
    parser.add_argument(
      '--test_ratio',
      type=float,
      default=0.3,
      help='ratio of data shoudl use to test.'
    )
    parser.add_argument(
      '--model_name',
      type=str,
      default='test',
      help='a name for the nodel- test and train runs have to use the same name.'
    )
    parser.add_argument(
      '--num_epochs',
      type=int,
      default=40,
      help='number of epochs to train on'
    )
    parser.add_argument(
      '--embedding_dim',
      type=int,
      default=100,
      help='number of dimensions to use, for glove valid values ate 50,100,200,300'
    )
    parser.add_argument(
      '--max_sequence_length',
      type=int,
      default=1000,
      help='The max length we will use for our model, text longer will be chopped. Text shorter will be padded.'
    )
    parser.add_argument(
      '--rnd_seed',
      type=int,
      default=randint(0,4294967295),
      help='The rnd seed to use.'
    )
    parser.add_argument(
      '--max_num_words',
      type=int,
      default=20000,
      help='The maximum number of words to be used.'
    )
    parser.add_argument(
      '--dropout_ratio',
      type=float,
      default=0.5,
      help='The ratio of dropout in LSTM layer. User larger dropouts to prevent overfitting.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    print('FLAGS:')
    print(FLAGS)
    #sys.exit()
    print ("random seed "+ str(FLAGS.rnd_seed))
    np.random.seed(FLAGS.rnd_seed)
    if FLAGS.mode == 'train':
        acc, num = train(FLAGS.model_name,FLAGS.text_dir,FLAGS.test_ratio,FLAGS.num_epochs,FLAGS.dropout_ratio,FLAGS.embedding_dim,FLAGS.glove_dir,FLAGS.max_sequence_length,FLAGS.max_num_words)
        print ('validation accuracy '+str(acc))
    else:
        results = test(FLAGS.model_name,FLAGS.text_dir,FLAGS.max_sequence_length)
        print (results)

def setup_tf_session():
    # just use the memory need on GPU card
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    KTF.set_session(sess)

def test(model_name,test_dir,max_sequence_length,models_dir='../models/'):
    model = load_model(model_name,models_dir)
    labels = load_labels(model_name,models_dir)
    data, names = load_tokenize_text(test_dir,model_name,max_sequence_length,models_dir)
    preds = model.predict(data)
    i = 0
    vals = {}
    for pred in preds:
        results = {}
        for j in range(0,len(pred)):
            results[labels[j]] = pred[j]
        vals[names[i]] = results
        i += 1
    return vals

def load_tokenize_text(test_dir,model_name,max_sequence_length,dir):
    texts, names = load_test_texts(test_dir)
    print('Found %s texts.' % len(texts))
    tokenizer = pickle_to_obj(os.path.join(dir,model_name+'_tokenizer.pkl'))
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=max_sequence_length)
    return data, names

def train(model_name,text_data_dir,test_split,num_epochs,dropout,embedding_dim,glove_dir,max_sequence_length,max_num_words,model_dir='../models/'):
    no_glove = False
    if glove_dir == None:
        no_glove = True
        embeddings_index = {}
    else:
        embeddings_index = load_embeddings(embedding_dim,glove_dir)
    texts, labels_index, labels = load_texts(text_data_dir)
    pickle_to_file(labels_index,os.path.join(model_dir,model_name+'_labels.pkl'))
    print('Found %s texts.' % len(texts))

    data, word_index = tokenize_data(texts,model_name,max_sequence_length)
    print(data)
    labels_cat = to_categorical(np.asarray(labels))
    x_train, y_train, labels_train, x_test, y_test, labels_test = split_data(data,labels_cat,labels,test_split)
    model = create_model(labels_index, word_index, embeddings_index,embedding_dim,dropout,max_sequence_length,max_num_words,no_glove)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    model.fit(x_train, y_train, validation_data=(x_test, y_test),
              nb_epoch=num_epochs, batch_size=128)

    save_model(model,model_name,model_dir)
    score, acc = model.evaluate(x_test, y_test, batch_size=128)
    print('Score: %1.4f' % score)
    print('Accuracy: %1.4f' % acc)
    
    pred_train = model.predict(x_train)
    pred_val = model.predict(x_test)
    labels_index_arr = np.asarray(labels_index.keys())[np.asarray(labels_index.values()).argsort()]    
    inv_labels_index_dict = {v: k for k, v in labels_index.items()}
    df_train = pd.DataFrame(pred_train, columns=labels_index_arr)
    df_train['TVH'] = 'T'
    df_train['actual'] = [inv_labels_index_dict[x] for x in labels_train]
    df_val = pd.DataFrame(pred_val, columns=labels_index_arr)
    df_val['TVH'] = 'V'
    df_val['actual'] = [inv_labels_index_dict[x] for x in labels_test]
    df_tv = pd.concat([df_train, df_val])
    df_tv.to_csv(os.path.join('../models', FLAGS.model_name + '_scored.csv'))
    return acc, len(x_train)

def pickle_to_file(which,filename):
    print(filename)
    f = open(filename, 'wb')   
    pickle.dump(which, f)
    f.close()    

def pickle_to_obj(filename):
    with open(filename, "rb") as input_file:
        new_obj = pickle.load(input_file)
    return new_obj

def load_labels(name,dir):
    file_name = name+'_labels.pkl'
    reverse_labels = pickle_to_obj(os.path.join(dir,file_name))
    labels = []
    for i in range(0,len(reverse_labels)):
        labels.append(reverse_labels.keys()[reverse_labels.values().index(i)])
    return labels

def save_model(model,name,dir):
    model_json = model.to_json()
    with open(dir+'/'+name+'.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(dir,name+'.h5'))
    print("Saved model to disk")

def load_model(name,dir='models/'):
    json_file = open(os.path.join(dir,name+'.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join(dir,name+'.h5'))
    print("Loaded model from disk")
    return loaded_model

def load_embeddings(embedding_dims,glove_dir):
    print('Indexing word vectors.')
    embeddings_index = {}
    file_name = 'glove.6B.100d.txt'
    if(embedding_dims  == 50):
        file_name = 'glove.6B.50d.txt'
    elif(embedding_dims  == 100):
        file_name = 'glove.6B.100d.txt'
    elif(embedding_dims  == 200):
        file_name = 'glove.6B.200d.txt'
    elif(embedding_dims  == 300):
        file_name = 'glove.6B.300d.txt'
    f = open(os.path.join(glove_dir, file_name))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def load_texts(data_dir):
    texts = []
    labels_index = {} 
    labels = [] 
    for name in sorted(os.listdir(data_dir)): # each file under the main dir
	path = os.path.join(data_dir, name)
        if os.path.isdir(path): # only walk through dirs
            label_id = len(labels_index)
            labels_index[name] = label_id
	    for fname in sorted(os.listdir(path)):
                #texts.append(read_text_file(os.path.join(path, fname, fname+'.txt')))
                texts.append(read_text_file(os.path.join(path, fname, 'output.txt')))
		labels.append(label_id)
    return texts, labels_index, labels

def load_test_texts(path):
    texts = []
    names = []
    for fname in sorted(os.listdir(path)):
        texts.append(read_text_file(os.path.join(path, fname, fname+'.txt')))
        names.append(fname)
    return texts, names

def read_text_file(path):
    if sys.version_info < (3,):
        f = open(path)
    else:
        f = open(path, encoding='latin-1')
    text = f.read()
    f.close()
    return text

def tokenize_data(texts,model_name,max_num_words,dir='../models/'):
    tokenizer = Tokenizer(nb_words=max_num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    file_name = model_name+'_tokenizer.pkl'
    pickle_to_file(tokenizer,os.path.join(dir,file_name))
    data = pad_sequences(sequences, maxlen=max_num_words)
    return data, word_index

def split_data(data,labels_cat,labels,split_ratio):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels_cat = labels_cat[indices]
    labels = np.asarray(labels)[indices]
    nb_test_samples = int(split_ratio * data.shape[0])

    x_train = data[:-nb_test_samples]
    y_train = labels_cat[:-nb_test_samples]
    labels_train = labels[:-nb_test_samples]
    x_test = data[-nb_test_samples:]
    y_test = labels_cat[-nb_test_samples:]
    labels_test = labels[-nb_test_samples:]
    return x_train, y_train, labels_train, x_test, y_test, labels_test

def create_embedding_layer(nb_words,word_index,embeddings_index,embedding_dim,max_sequence_length,max_num_words,no_glove=False):
    if no_glove == False:
        embedding_matrix = np.zeros((nb_words, embedding_dim))
        for word, i in word_index.items():
            if i >= max_num_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            embedding_layer = Embedding(nb_words,
                                        embedding_dim,
                                        weights=[embedding_matrix],
                                        input_length=max_sequence_length,
                                        trainable=False)
    else:
        print('Not using glove')
        embedding_layer = Embedding(len(word_index) + 1,
                            embedding_dim,
                            input_length=max_sequence_length)
    return embedding_layer

def create_model(labels_index, word_index, embeddings_index,embedding_dim,dropout,max_sequence_length,max_num_words,no_glove):
    nb_words = min(max_num_words, len(word_index)+1)
    print ('nb_words '+str(nb_words))
    embedding_layer = create_embedding_layer(nb_words, word_index, embeddings_index,embedding_dim, max_sequence_length,max_num_words,no_glove)
    if (len(labels_index) < (MAX_POOL_SIZE-1)):
        p_size = len(labels_index) + 1
    else:
        p_size = MAX_POOL_SIZE
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(CNN_SIZE, p_size, activation='relu')(embedded_sequences)
    x = MaxPooling1D(p_size)(x)
    x = Conv1D(CNN_SIZE, p_size, activation='relu')(x)
    x = MaxPooling1D(pool_length=2)(x)
    x = LSTM(LSTM_SIZE)(x)
    x = Dropout(dropout)(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(len(labels_index), activation='softmax')(x)
    model = Model(sequence_input, preds)
    print(model.summary())
    return model

if __name__ == "__main__":
    main()

