from flask import Flask, request, jsonify
import util
app = Flask(__name__)


@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"


@app.route('/get_prediction/', methods=['POST'])
def get_prediction():
    text = str(request.form['text'])
    response = jsonify({
        'prediction': util.prediction(text)
    })
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


if __name__ == '__main__':
    util.load_save_model()
    app.run()
