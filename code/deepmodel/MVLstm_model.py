from keras.layers import InputSpec, Layer, Input, Dense, merge, Conv1D
from keras.layers import Lambda, Activation, Dropout, Embedding, TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
import numpy as np 
import math
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from Match import *

class MVLSTM:
    @staticmethod
    def build_model(emb_matrix, max_sequence_length):
        hidden_size = 32
        topk = 100
        dropout_rate = 0.4
        
        # The embedding layer containing the word vectors
        emb_layer = Embedding(
            input_dim=emb_matrix.shape[0],
            output_dim=emb_matrix.shape[1],
            weights=[emb_matrix],
            input_length=max_sequence_length,
            trainable=False)

        # Define inputs
        seq1 = Input(shape=(max_sequence_length,))
        seq2 = Input(shape=(max_sequence_length,))
        # Run inputs through embedding
        q_embed = emb_layer(seq1)
        d_embed = emb_layer(seq2)

        q_rep = Bidirectional(LSTM(hidden_size, return_sequences=True, dropout=dropout_rate))(q_embed)
        d_rep = Bidirectional(LSTM(hidden_size, return_sequences=True, dropout=dropout_rate))(d_embed)
 
        cross = Match(match_type='dot')([q_rep, d_rep])
        cross_reshape = Reshape((-1, ))(cross)
        mm_k = Lambda(lambda x: K.tf.nn.top_k(x, k=topk, sorted=True)[0])(cross_reshape)
        pool1_flat_drop = Dropout(rate=dropout_rate)(mm_k)
        dense = Dense(30,activation='relu')(pool1_flat_drop)
        dense = Dropout(rate=dropout_rate)(dense)
        dense = Dense(6,activation='relu')(dense)
        dense = Dropout(rate=dropout_rate)(dense)
        pred = Dense(1, activation='sigmoid')(dense)
        model = Model(inputs=[seq1, seq2], outputs=pred)
        ada = Adam(lr=0.001)
        model.compile(loss='binary_crossentropy', optimizer=ada, metrics=['acc'])

        return model

