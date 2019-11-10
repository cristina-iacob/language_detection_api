from flask import Flask
from flask import request
from flask import render_template
from flask import send_from_directory
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout

import numpy as np
import pandas as pd
import os

from tensorflow.keras.models import load_model

max_letters=12
char_count=104
char_count2=26

label_names=['english', 'french', 'german', 'romanian']

network = Sequential()
network.add(Dense(200, input_dim=(char_count*max_letters)-1, activation='sigmoid'))
network.add(Dense(150, activation='sigmoid'))
network.add(Dense(100, activation='sigmoid'))
network.add(Dense(100, activation='sigmoid'))
network.add(Dense(len(label_names), activation='softmax'))

network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

network = load_model("./lang_detect.hdf5")

# network3 = Sequential()
# network3.add(Dense(512, input_dim=(char_count*max_letters)-1))
# network3.add(Activation('relu'))
# network3.add(Dropout(0.5))
# network3.add(Dense(512, activation='sigmoid'))
# network3.add(Dropout(0.4))
# network3.add(Dense(512, activation='sigmoid'))
# network3.add(Dropout(0.3))
# network3.add(Dense(len(label_names), activation='softmax'))

# network3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# network3 = load_model("./lang_detect_longer.hdf5")

network2 = Sequential()
network2.add(Dense(512, input_dim=(char_count2*max_letters)-1))
network2.add(Activation('relu'))
network2.add(Dropout(0.5))
network2.add(Dense(512, activation='sigmoid'))
network2.add(Dropout(0.4))
network2.add(Dense(512, activation='sigmoid'))
network2.add(Dropout(0.3))
network2.add(Dense(len(label_names), activation='softmax'))

network2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

network2 = load_model("./lang_detect_n2.hdf5")

def convert_dic_to_vector(dic, max_word_length):
    new_list = []
    for word in dic:
        vec = ''
        n = len(word)
        for i in range(n):
            current_letter = word[i]
            ind = ord(current_letter)-97
            #ind = ord(current_letter)
            placeholder = (str(0)*ind) + str(1) + str(0)*((char_count-1)-ind)
            vec = vec + placeholder
        if n < max_word_length:
            excess = max_word_length-n
            vec = vec +str(0)*char_count*excess
        new_list.append(vec)
   # print(len(new_list))
    return new_list

def predict_word(word):
    dic = []
    guess = []
    guess2 = []

    dic.append(word)
    vct_str = convert_dic_to_vector(dic, max_letters-1)
    vct = np.zeros((1, (char_count * max_letters)-1))
    vct2 = np.zeros((1, (char_count2 * max_letters)-1))
    count = 0
    #print(len(vct_str[0]))
    for digit in vct_str[0]:
        vct[0,count] = int(digit)
        count += 1
    prediction_vct = network.predict(vct)
    prediction2_vct = network2.predict(vct2)
    prediction_winner = network.predict_classes(vct)
    prediction_winner2 = network2.predict_classes(vct2)

    langs = list(label_names)


    for i in range(len(label_names)):
        lang = langs[i]
        winner=0
        winner2=0
        score = prediction_vct[0][i]
        score2 = prediction2_vct[0][i]
        if i == prediction_winner[0]:
            winner=1
        if i == prediction_winner2[0]:
            winner2=1
        
        #guess["language"+str(i)]=lang
        #guess["confidence"+str(i)]=str(round(100*score, 2)) + '%'

        guess.append({
            "winner" : winner,
            "word": word, 
            "language": lang, 
            "confidence":round(100*score, 1),
            })

        guess2.append({
            "winner" : winner2,
            "word": word, 
            "language": lang, 
            "confidence":round(100*score2, 1),
            })

        print(prediction_winner)    
        print(prediction_winner2)
        #print(lang + ': ' + str(round(100*score, 2)) + '%')

    #print(prediction_winner[0])
    return guess, guess2

# Flask Setup
app = Flask(__name__, static_url_path='')

################# Flask Routes ###################
@app.route('/', methods=['GET', 'POST'])
def root():
    if request.method == 'POST':
        word = request.form.get("word","")
        robots=predict_word(word)
        return render_template('index.html', robots=robots, word=word)
    
    return render_template('index.html', robots="")

@app.route('/about')
def about():
	return app.send_static_file('about.html')

if __name__ == "__main__":
    app.run(debug=True)