from flask import Flask
from flask import request
from flask import render_template
from flask import send_from_directory

import pandas as pd
import os

# Test FLASK with premade model
from tensorflow.keras.models import load_model
model_1 = load_model("./voice_model_trained.h5")

from sklearn.linear_model import LogisticRegression

voice = pd.read_csv(os.path.join('.','voice.csv'))
X = voice.drop("label", axis=1)
y = voice["label"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)

classifier = LogisticRegression()
classifier
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test.head(1))

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