from os import listdir
from os.path import isfile, join
import pandas as pd
import re
import uvicorn
from pydantic import BaseModel
import json
import requests
import urllib.request
import boto3
from io import StringIO

# Change you credentials
ACCESS_KEY = ''
SECRET_KEY = ''
BUCKET_NAME = ''

client = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

# Reading the csv
df = pd.read_csv('') # Location of the csv to be given
statement = []
prediction = []

# For every company present in the csv
for i in range(len(df)):
    company = df.iloc[i]['Company']
    year = df.iloc[i]['Year']

    # Connecting to FastAPI to get the transcript
    URL = "http://127.0.0.1:7070/call-transcripts/{}/2021".format(company)
    file = urllib.request.urlopen(URL)
    for line in file:
        # Reading the transcript
        decoded_line = line.decode("utf-8")
        txt = decoded_line.replace('\\n', '\n')

        # Removing the following words from file
        text = txt.replace('Operator', '')
        text = text.replace('[Operator Instructions]', '')
        text = text.replace('Company Participants', '')
        text = text.replace('[ Instructions]', '')
        text = text.replace('Unidentified Analyst', '')
        text = text.replace('Company Participants', '')
        text = text.replace('Conference Call Participants', '')

        # Keeping only Alphabets, numbers and space
        text = re.sub('[^a-zA-Z0-9., \n]', '', text)

        # Splitting the line
        sentences = (text.splitlines())

        # Removing the spaces
        sentences = [x for x in sentences if x != '']

        # Converting the statements to json format to pass to Flask
        output = dict()
        output["data"] = sentences
        app_json = json.dumps(output)
        headers = {'Content-type': 'application/json', }

        # Connecting to flask and getting the response
        flask_con = requests.post('http://0.0.0.0:5000/predict', json=app_json)
        flask_out = (flask_con.json())

        # Getting the sentence and prediction and appending it to the lists
        sent = flask_out['input']['data']
        val = flask_out['pred']
        for i in val:
            i = str(i).replace('[', '').replace(']', '')
            if float(i) > 0.5:
                pred = 'positive'
            elif float(i) == 0:
                pred = 'neutral'
            else:
                pred = 'negative'
            prediction.append(pred)
        for i in sent:
            statement.append(i)

print('Upload to S3')
# Creating a dataframe with the sentence and prediction
prediction_df = pd.DataFrame()
prediction_df['Statement'] = statement
prediction_df['Prediction'] = prediction

# Converting the dataframe to csv and storing it in S3 bucket
prediction_final = StringIO()
prediction_df.to_csv(prediction_final, header=True, index=False)
prediction_final.seek(0)
client.put_object(Bucket=BUCKET_NAME, Body=prediction_final.getvalue(), Key='change to location in s3') 

print('The End')

