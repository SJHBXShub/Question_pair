from keras.layers import InputSpec, Layer, Input, Dense, merge, Conv1D, Dot, Permute, Multiply
from keras.layers import Lambda, Activation, Dropout, Embedding, TimeDistributed
from keras.layers.core import SpatialDropout1D
from keras.layers.recurrent import LSTM, GRU
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
import keras.backend as K
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
import numpy as np 
import math
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint

#Decomposable Attention Model by Ankur P. Parikh et al. 2016

class DA:
    @staticmethod
    def build_model(emb_matrix, max_sequence_length):
        ############# Embedding Process ############
        # The embedding layer containing the word vectors
        emb_layer = Embedding(
            input_dim=emb_matrix.shape[0],
            output_dim=emb_matrix.shape[1],
            weights=[emb_matrix],
            input_length=max_sequence_length,
            trainable=False
        )
        # attention model ##########
        # Define inputs
        seq1 = Input(shape=(max_sequence_length,))
        seq2 = Input(shape=(max_sequence_length,))

        # Run inputs through embedding
        emb1 = emb_layer(seq1)
        emb2 = emb_layer(seq2)
        
        emb_distributed = TimeDistributed(Dense(200,
                                   activation='relu',
                                   kernel_regularizer=l2(1e-5),
                                   bias_regularizer=l2(1e-5)))
        
        emb1 = Dropout(0.4)(emb_distributed(emb1))
        emb2 = Dropout(0.4)(emb_distributed(emb2))
        
        # score each words and calculate score matrix
        F_seq1, F_seq2 = emb1, emb2
        for i in range(2):
            scoreF = TimeDistributed(Dense(200,
                                     activation='relu',
                                     kernel_regularizer=l2(1e-5),
                                     bias_regularizer=l2(1e-5)))
            F_seq1 = Dropout(0.4)(scoreF(F_seq1))
            F_seq2 = Dropout(0.4)(scoreF(F_seq2))
        cross = Dot(axes=(2,2))([F_seq1, F_seq2])
        
        # normalize score matrix, encoder premesis and get alignment
        c1 = Lambda(lambda x: keras.activations.softmax(x))(cross)
        c2 = Permute((2,1))(cross)
        c2 = Lambda(lambda x: keras.activations.softmax(x))(c2)
        seq1Align = Dot((2,1))([c1,emb2])
        seq2Align = Dot((2,1))([c2,emb1])
        
        # Concat original and alignment, score each pair of alignment        
        seq1Align = concatenate([emb1,seq1Align])
        seq2Align = concatenate([emb2,seq2Align])
        for i in range(2):
            scoreG = TimeDistributed(Dense(200,
                                     activation='relu',
                                     kernel_regularizer=l2(1e-5),
                                     bias_regularizer=l2(1e-5)))
            seq1Align = scoreG(seq1Align)
            seq2Align = scoreG(seq2Align)
            seq1Align = Dropout(0.4)(seq1Align)
            seq2Align = Dropout(0.4)(seq2Align)
        
        # Sum all these scores, and make final judge according to sumed-score
        sumwords = Lambda(lambda x: K.reshape(K.sum(x, axis=1, keepdims=True), (-1, 200)))
        V_seq1 = sumwords(seq1Align)
        V_seq2 = sumwords(seq2Align)
        final = concatenate([V_seq1, V_seq2])
        for i in range(2):
            final = Dense(200,
                    activation='relu',
                    kernel_regularizer=l2(1e-5),
                    bias_regularizer=l2(1e-5))(final)
            final = Dropout(0.4)(final)
            final = BatchNormalization()(final)

        pred = Dense(1, activation='sigmoid')(final)

        # model = Model(inputs=[seq1, seq2, magic_input, distance_input], outputs=pred)
        model = Model(inputs=[seq1, seq2], outputs=pred)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
        print (model.summary())
        return model