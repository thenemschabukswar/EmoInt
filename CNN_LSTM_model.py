from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding , LSTM
from tensorflow.keras.layers import Dense , Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import Activation
from keras.layers.convolutional import Convolution1D
#import theano
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




features = 20000
max_sentence_len = 50
embeddings_dim = 300
seed = 27
crossval = 20
filter_length = 3
pool_size = 2
reg_dimensions = 1

#Create an empty dictionary
embeddings = dict()

#Load external word embeddings
embeddings = KeyedVectors.load_word2vec_format(datapath( "twitter_sgns_subset.txt.gz") , binary=False )

full_voc_data = []
with open("vocab.csv", errors="ignore") as f:
    data = f.read()
    lines = data.split('\n')
    for line in lines:  
        temp = line.split('\t')
        full_voc_data.append((temp[1],float(temp[3])))
    

full_data_size = int(len(full_voc_data))
all_texts = [ txt for ( txt, label ) in full_voc_data[0:full_data_size] ]

tokenizer = Tokenizer(num_words=features, filters='%&()*+,-./:;<=>[\\]^_`{|}~\t\n',lower=True, split=" ")
tokenizer.fit_on_texts(all_texts)


TSdata = []

with open("test/anger_CNN-LSTM_input.txt", errors="ignore") as f:
    data = f.read()
    lines = data.split('\n')
    for line in lines:  
        temp = line.split('\t')
        TSdata.append((temp[0],float(temp[1])))

#TSdata = [ ( row["Tweet"] , float( row["Rating"] )  ) for row in csv.DictReader(open("cnn_lstm/test/anger_CNN-LSTM_input.txt"), delimiter='\t', quoting=csv.QUOTE_NONE) ]
test_size = int(len(TSdata) )
test_texts = [ txt for ( txt, label ) in TSdata[0:test_size] ]
test_labels = [ np.asarray(label) for ( txt , label ) in TSdata[0:test_size] ]
test_sequences = sequence.pad_sequences( tokenizer.texts_to_sequences( test_texts ) , maxlen=max_sentence_len )

EMOS="joy fear sadness anger afjs"

for i in EMOS.split():
    currentemo=str(i)
    TRdata = []
    with open("train/"+currentemo+"_tr_dv.csv", errors="ignore") as f:
        data = f.read()
        lines = data.split('\n')
        for line in lines:  
            temp = line.split('\t')
            TRdata.append((temp[1],float(temp[3])))
    random.shuffle( TRdata )    
    train_size = int(len(TRdata) )
    train_texts = [ txt for ( txt, label ) in TRdata[0:train_size] ]
    train_labels = [ np.asarray(label) for ( txt , label ) in TRdata[0:train_size] ]
    train_sequences = sequence.pad_sequences( tokenizer.texts_to_sequences( train_texts ) , maxlen=max_sentence_len )
    embedding_weights = np.zeros( ( features , embeddings_dim ) )
    for word,index in tokenizer.word_index.items():
        if index < features:
            try: embedding_weights[index,:] = embeddings[word]
            except: embedding_weights[index,:] = np.random.rand( 1 , embeddings_dim )
    np.random.seed(seed)
    filter_length = 3
    nb_filter = embeddings_dim
    model = Sequential()
    model.add(Embedding(features, embeddings_dim, input_length=max_sentence_len, weights=[embedding_weights]))
    model.add(Dropout(0.25))
    model.add(Convolution1D( nb_filter, filter_length, activation='relu'))
    model.add(MaxPool1D(pool_size=pool_size))
    model.add(LSTM(embeddings_dim))
    model.add(Dense(reg_dimensions))
    model.add(Activation('sigmoid'))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    model.fit( train_sequences , train_labels , epochs=10)
    model.save("models/"+currentemo+".h5")  # creates a HDF5 file 'my_model.h5'
    results = model.predict( test_sequences )
    np.savetxt("test/pred/"+currentemo+".txt", results, newline='\n')
    
    