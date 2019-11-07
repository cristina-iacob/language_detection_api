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
label_names=['english', 'french', 'german', 'romanian']

network = Sequential()
network.add(Dense(200, input_dim=(char_count*max_letters)-1, activation='sigmoid'))
network.add(Dense(150, activation='sigmoid'))
network.add(Dense(100, activation='sigmoid'))
network.add(Dense(100, activation='sigmoid'))
network.add(Dense(len(label_names), activation='softmax'))

network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

network = load_model("./lang_detect.hdf5")

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
    print(len(new_list))
    return new_list


dic = []
dic.append("hello")
vct_str = convert_dic_to_vector(dic, max_letters-1)
vct = np.zeros((1, (char_count * max_letters)-1))
count = 0
print(len(vct_str[0]))
for digit in vct_str[0]:
    vct[0,count] = int(digit)
    count += 1
prediction_vct = network.predict(vct)

langs = list(label_names)
for i in range(len(label_names)):
    lang = langs[i]
    score = prediction_vct[0][i]
    print(lang + ': ' + str(round(100*score, 2)) + '%')

# Flask Setup
app = Flask(__name__, static_url_path='')

def what_lang(word):
    guesses=[
{"language": "German", "algorithm": f"{word} {predictions}" },
{"language": "French", "algorithm": f"{word} {predictions}" },
{"language": "Romanian", "algorithm": f"{word} {predictions}" }
]

    robots_l=[]

    for guess in guesses:
        robots_l.append(guess)
        print (robots_l)
    
    return robots_l

################# Flask Routes ###################
@app.route('/', methods=['GET', 'POST'])
def root():
    if request.method == 'POST':
        word = request.form.get("word","")
        robots=what_lang(word)
        return render_template('index.html', robots=robots)
    
    return render_template('index.html', robots="")

@app.route('/about')
def about():
	return app.send_static_file('about.html')

if __name__ == "__main__":
    app.run(debug=True)