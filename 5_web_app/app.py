from flask import Flask, render_template, request
import bleach
from predict import get_prediction

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    title = bleach.clean(request.form['title'])
    author = bleach.clean(request.form['author'])
    body = bleach.clean(request.form['body'])
    text = f"{title} {author} {body}"
    prediction, time = get_prediction(text)
    return render_template('result.html', prediction=prediction, time=f"{time:.2f}" )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5000)