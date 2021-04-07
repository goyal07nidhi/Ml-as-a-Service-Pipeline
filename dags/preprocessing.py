import glob
import re
import os
import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions
import boto3
import pandas as pd
from io import StringIO

# Change your access keys
ACCESS_KEY = 'access_key'
SECRET_KEY = 'aws_access_secret_key'
BUCKET = 'bucket_name'

# Connect to Boto3
s3 = boto3.resource(
    service_name='s3',
    region_name='us-east-2',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY)

client = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

# Enter your S3 Bucket name here


# Enter you filepath
file_path = '/sec-edgar/call_transcripts'


def convert_to_txt():
    for filename in glob.iglob(file_path + '*'):
        os.rename(filename, filename + '.txt')
    print('Conversion Complete')


def upload_to_s3():
    for filename in glob.iglob(file_path + '*'):
        file_name = os.path.basename(filename)
        s3.Bucket(bucket_name).upload_file(Filename=file_path + file_name, Key='sec-edgar/transcripts/' + file_name)
    print('Upload Complete')


def ibm_api():
    result = client.list_objects(Bucket=bucket_name, Prefix='sec-edgar/transcripts/', Delimiter='/')
    # print(result)
    text_append = []
    for i in result.get('Contents'):
        data = client.get_object(Bucket=bucket_name, Key=i['Key'])
        contents = data['Body'].read().decode('utf-8')

        # remove Operator and [Operator Operations]
        text = contents.replace('Operator', '')
        text = text.replace('[Operator Instructions]', '')
        text = text.replace('Company Participants', '')
        text = text.replace('[ Instructions]', '')
        text = text.replace('Unidentified Analyst', '')
        text = text.replace('Company Participants', '')
        text = text.replace('Conference Call Participants', '')

        # Keeping alphabets, numbers and spaces
        text = re.sub('[^a-zA-Z0-9., \n]', '', text)

        # Splitting them based on lines
        text_list = text.splitlines()

        # Removing empty space values
        text_list = [x for x in text_list if x != '']

        # Appending the final text
        text_append.append(text_list)

    text_list = [item for sublist in text_append for item in sublist]
    text_file = set(text_list)

    authenticator = IAMAuthenticator('U5W5wzkqMrYckmBB3LvaWyEBzzonaDdMkT_z2ZtaDlAb')

    natural_language_understanding = NaturalLanguageUnderstandingV1(
        version='2020-08-01',
        authenticator=authenticator
    )

    l_text = []
    l_score = []
    l_label = []

    df = pd.DataFrame()

    for item in text_file:
        try:
            response = natural_language_understanding.analyze(text=item, features=Features(
                sentiment=SentimentOptions(targets=[item])), language="en").get_result()
            x = json.dumps(response)
            x_load = json.loads(x)
            l_text.append(x_load["sentiment"]["targets"][0]["text"])
            l_score.append(x_load["sentiment"]["targets"][0]["score"])
            l_label.append(x_load["sentiment"]["targets"][0]["label"])

        except:
            pass

    df['text'] = l_text
    df['score'] = l_score
    df['label'] = l_label

    # Uploading to S3
    text_final = StringIO()
    df.to_csv(text_final, header=True, index=False)
    text_final.seek(0)
    client.put_object(Bucket=bucket_name, Body=text_final.getvalue(), Key='sec-edgar/annotation/text_final.csv')
