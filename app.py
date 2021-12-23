from flask import Flask, render_template, request
import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import datetime
import random
import nltk.translate.bleu_score as bleu
from nltk.translate.bleu_score import SmoothingFunction



def decontractions(phrase):
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"won\â€™t", "will not", phrase)
    phrase = re.sub(r"can\â€™t", "can not", phrase)

    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    phrase = re.sub(r"n\â€™t", " not", phrase)
    phrase = re.sub(r"\â€™re", " are", phrase)
    phrase = re.sub(r"\â€™s", " is", phrase)
    phrase = re.sub(r"\â€™d", " would", phrase)
    phrase = re.sub(r"\â€™ll", " will", phrase)
    phrase = re.sub(r"\â€™t", " not", phrase)
    phrase = re.sub(r"\â€™ve", " have", phrase)
    phrase = re.sub(r"\â€™m", " am", phrase)

    return phrase

def preprocess(text):
    # convert all the text into lower letters
    # use this function to remove the contractions: https://gist.github.com/anandborad/d410a49a493b56dace4f814ab5325bbd
    # remove all the spacial characters: except space ' '
    text = text.lower()
    text = decontractions(text)
    text = re.sub('[^A-Za-z0-9 ]+', '', text)
    return text

def preprocess_ita(text):
    text = text.lower()
    text = decontractions(text)
    text = re.sub('[$)\?"â€™.Â°!;\'â‚¬%:,(/]', '', text)
    text = re.sub('\u200b', ' ', text)
    text = re.sub('\xa0', ' ', text)
    text = re.sub('-', ' ', text)
    return text

class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, input_length):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.enc_units= enc_units
        self.lstm_output = 0
        self.lstm_state_h=[]
        self.lstm_state_c=[]

    def build(self, input_shape):
        self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.input_length,
                           mask_zero=True, name="embedding_layer_encoder")
        self.lstm = LSTM(self.enc_units, return_state=True, return_sequences=True, name="Encoder_LSTM", activation='sigmoid')

    def call(self, input_sentances, training=True):
        input_embedd                           = self.embedding(input_sentances)
        self.lstm_output, self.lstm_state_h,self.lstm_state_c = self.lstm(input_embedd)
        return self.lstm_output, self.lstm_state_h,self.lstm_state_c
    def get_states(self):
        return self.lstm_state_h,self.lstm_state_c
    def initialize_states(self,batch_size):
        for i in range(batch_size):
            self.lstm_state_h.append([0]*self.enc_units)
            self.lstm_state_c.append([0]*self.enc_units)
        self.lstm_state_h = np.array(self.lstm_state_h)
        self.lstm_state_c = np.array(self.lstm_state_c)
        return self.lstm_state_h,self.lstm_state_c



class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, dec_units, input_length):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = 100
        self.dec_units = dec_units
        self.input_length = input_length
        # we are using embedding_matrix and not training the embedding layer
        self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.input_length,
                           mask_zero=True, name="embedding_layer_decoder", trainable=True)
        self.lstm = LSTM(self.dec_units, return_sequences=True, return_state=True, name="Encoder_LSTM", activation='sigmoid')

    def call(self, target_sentances, states):
        state_h = states[0]
        state_c = states[1]
        target_embedd           = self.embedding(target_sentances)
        lstm_output, decoder_final_state_h, decoder_final_state_c        = self.lstm(target_embedd, initial_state=[state_h, state_c])
        return lstm_output, decoder_final_state_h, decoder_final_state_c

class MyModel(Model):
    def __init__(self, encoder_inputs_length,decoder_inputs_length, output_vocab_size):
        super().__init__() # https://stackoverflow.com/a/27134600/4084039
        self.encoder = Encoder(vocab_size=vocab_size_ita+1, embedding_dim=50, enc_units=256, input_length=encoder_inputs_length)
        self.decoder = Decoder(vocab_size=vocab_size_eng+1, embedding_dim=100, dec_units=256, input_length=decoder_inputs_length)
        self.dense   = Dense(output_vocab_size+1, activation='softmax')


    def call(self, data):
        input,output = data[0], data[1]
        encoder_output, encoder_h, encoder_c = self.encoder(input)
        decoder_output,_,_                       = self.decoder(output, [encoder_h, encoder_c])
        output                               = self.dense(decoder_output)
        return output
    
    

fp = open('tk_ita','rb')
tknizer_ita = pickle.load(fp)
fp.close()
fp = open('tk_eng','rb')
tknizer_eng = pickle.load(fp)
fp.close()
vocab_size_eng=len(tknizer_eng.word_index.keys())

vocab_size_ita=len(tknizer_ita.word_index.keys())



def predict_mod1(source_sentence):
    model  = MyModel(encoder_inputs_length=20,decoder_inputs_length=20,output_vocab_size=vocab_size_eng)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy')

    model.load_weights('cool')
    
    inputs = [tknizer_ita.word_index[i] for i in source_sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=20, padding='post')
    enc_inp = tf.convert_to_tensor(inputs)
  
    result = ''
    enc_out, enc_h, enc_c = model.encoder(enc_inp)
    states = [enc_h,enc_c]
    dec_input = tf.expand_dims([tknizer_eng.word_index['<start>']], 0)
    for _ in range(20):
        dec_out,state_h,state_c=model.decoder(dec_input,states)
        dense_out = model.layers[2](dec_out)
        predicted_id = tf.argmax(dense_out[0][0]).numpy()
        if tknizer_eng.index_word[predicted_id] == '<end>':
          return result
        result += tknizer_eng.index_word[predicted_id] + ' '
        
  
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)
        states=[state_h,state_c]

    return result




app = Flask(__name__)


@app.route("/", methods = ["GET","POST"])
def start():
    predicted_sentence = ''
    if request.method == "POST":
        source_sentence = request.form["source"]
        
        processed_sentence = preprocess_ita(source_sentence)
        
        predicted_sentence = predict_mod1(processed_sentence)
        print(predicted_sentence)
        
    return render_template("index.html", pred = predicted_sentence)




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
