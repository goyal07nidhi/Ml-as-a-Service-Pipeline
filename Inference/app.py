from flask import Flask, request, jsonify
import boto3
import tensorflow_text as text
import tensorflow as tf
import json

app = Flask(__name__)


@app.route('/predict', methods=['GET', 'POST'])
def create_task():

    if request.method == "POST":
        sentences = request.get_json(force=True)
        if type(sentences) is str:
            sentence = json.loads(sentences)
        else:
            sentence = sentences
        output = dict()
        output["input"] = sentence
        output["pred"] = []

        checkpoint = load_model_on_keras()

        for s in sentence['data']:
            result = predict(checkpoint, s)
            output["pred"].append([result])

        return jsonify(output), 201

    if request.method == "GET":
        msg = f"Please compose your request in POST type with data."
        return jsonify({"msg": msg})


def get_model_from_s3():
    ACCESS_KEY = ''
    SECRET_KEY = ''
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)
    s3.download_file('bucket', 's3 file location', 'local file location') # change the required
    # return model_h5_file


def load_model_on_keras():
    checkpoint = tf.saved_model.load('location of the model') # change the path
    return checkpoint


def predict(checkpoint, sentence):
    x=[]
    x.append(sentence)
    metric_result = float(tf.sigmoid(checkpoint(tf.constant(x))))
    return metric_result


if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000", debug=True)
