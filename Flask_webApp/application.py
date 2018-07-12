
from flask import Flask, render_template, redirect
from flask_wtf import FlaskForm
from wtforms import Form, SubmitField, TextAreaField
import os
import string
import html
import webbrowser
import nltk
from nltk.corpus import stopwords as sw
import pickle


basepath = os.path.abspath(".")

app = Flask(__name__)

app.config.from_pyfile('default_config.py')



class FeatureForm(FlaskForm):
    review_text = TextAreaField(u'Review Text', render_kw={"rows": 5, "cols": 100})
    go = SubmitField('Submit')



class BaseModel:
    def __init__(self):
        self.lemmatizer = nltk.WordNetLemmatizer()

        self.model = None
        self.vec = None

    # Load Vec
    def load_vec(self, vec_path, mode='rb'):
        with open(vec_path, mode) as pkl_file:
            self.vec = pickle.load(pkl_file)

    # Load Model
    def load_model(self, model_path, mode='rb'):
        with open(model_path, mode) as pkl_file:
            self.model = pickle.load(pkl_file)

    # Preprocessing
    def preprocessing(self, line: str) -> str:
        stopwords = sw.words('english')
        stopwords = stopwords + ['not_' + w for w in stopwords]
        pad_punct = str.maketrans({key: " {0} ".format(key) for key in string.punctuation})
        invalidChars = str(string.punctuation.replace("_", ""))

        line = html.unescape(str(line))
        line = str(line).replace("can't", "can not")
        line = str(line).replace("n't", " not")
        line = str(line).translate(pad_punct)
        line = nltk.word_tokenize(line.lower())

        tokens = []
        negated = False
        for t in line:
            if t in ['not', 'no']:
                negated = not negated
            elif t in string.punctuation or not t.isalpha():
                negated = False
            else:
                tokens.append('not_' + t if negated else t)

        bi_tokens = list(nltk.bigrams(line))
        bi_tokens = list(map('_'.join, bi_tokens))
        bi_tokens = [i for i in bi_tokens if all(j not in invalidChars for j in i)]
        tri_tokens = list(nltk.trigrams(line))
        tri_tokens = list(map('_'.join, tri_tokens))
        tri_tokens = [i for i in tri_tokens if all(j not in invalidChars for j in i)]
        tokens = tokens + bi_tokens + tri_tokens

        tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in stopwords]
        return ' '.join(tokens)


    # Predict
    def predict(self, line):
        if self.model is None or self.vec is None:
            print('Modle / Vec is not loaded')
            return ""

        line = self.preprocessing(line)
        features = self.vec.transform([line])

        return self.model.predict(features)[0]



class ReviewModel(BaseModel):
    def __init__(self):
        super().__init__()

        self.load_vec(basepath + '/mysite/model/tf_vec.pkl')
        self.load_model(basepath + '/mysite/model/svc_model.pkl')

    def predict(self, line, highlight=True):
        sentiment = super(ReviewModel, self).predict(line)

        # highlight words, hack
        if highlight:
            highlight_words = [w for w in self.preprocessing(line).split()
                               if super(ReviewModel, self).predict(w) == sentiment]
            return sentiment, highlight_words
        else:
            return sentiment




@app.route('/', methods=('GET', 'POST'))
def index():
    myform = FeatureForm()

    if myform.is_submitted():
        line = myform.review_text.data
        review_model = ReviewModel()
        sentiment, hightwords = review_model.predict(line, highlight=True)

        return render_template('result.html',
                               line=line,
                               highlight_words=hightwords,
                               sentiment=sentiment)

    return render_template('welcome.html', form=myform)


@app.route('/result/')
def submit():
    return render_template('result.html')


@app.route('/about')
def about():
    return redirect("https://github.com/jesseqzhen/NLP_Sentiment_Analysis", code=302)

@app.route('/author')
def author():
    return redirect("https://www.linkedin.com/in/jesse-qz/", code=302)

if __name__ == "__main__":
    app.run()

