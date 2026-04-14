from flask import Flask, render_template, request
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))
# Initialize app
app = Flask(__name__)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

ps = PorterStemmer()

# Make sure required NLTK data is available
# nltk.download('punkt')
# nltk.download('stopwords')
import nltk
import os

nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)


def transform_text(text):
    if not text:
        return ""

    text = text.lower()

    try:
        words = nltk.word_tokenize(text)
    except:
        words = text.split()  # fallback if tokenizer fails

    words = [w for w in words if w.isalnum()]
    words = [w for w in words if w not in STOPWORDS]
    words = [ps.stem(w) for w in words]

    return " ".join(words)



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
            print("ERROR:", e)  # shows in Render logs
            prediction = str(e)  # show error on UI for debugging

    return render_template('index.html', prediction=prediction, message=input_sms)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)