from flask import Flask, render_template, request
import pickle
import string
import nltk
import os
from nltk.stem.porter import PorterStemmer

# ---------- APP INIT ----------
app = Flask(__name__)

# ---------- NLTK SETUP ----------
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download required data safely
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords', download_dir=nltk_data_path)

from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# ---------- MODEL LOAD ----------
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    print("MODEL LOAD ERROR:", e)

ps = PorterStemmer()

# ---------- TEXT PREPROCESS ----------
def transform_text(text):
    if not text:
        return ""

    text = text.lower()

    try:
        words = nltk.word_tokenize(text)
    except:
        words = text.split()  # fallback if tokenizer fails

    # remove non-alphanumeric
    words = [w for w in words if w.isalnum()]

    # remove stopwords
    words = [w for w in words if w not in STOPWORDS]

    # stemming
    words = [ps.stem(w) for w in words]

    return " ".join(words)

# ---------- ROUTE ----------
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    input_sms = ""

    if request.method == 'POST':
        input_sms = request.form.get('message', '')

        try:
            if input_sms.strip():
                transformed_sms = transform_text(input_sms)
                vector_input = tfidf.transform([transformed_sms])
                result = model.predict(vector_input)[0]

                prediction = "Spam" if result == 1 else "Not Spam"

        except Exception as e:
            print("ERROR:", e)
            prediction = "Something went wrong"

    return render_template('index.html', prediction=prediction, message=input_sms)

# ---------- RUN ----------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)