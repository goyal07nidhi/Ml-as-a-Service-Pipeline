# Imports the modules
import pandas as pd
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

# Import Custom Modules
import preprocessing as p


def push_models_to_s3():
    p.convert_to_txt()
    p.upload_to_s3()


def label_models():
    df = p.ibm_api()


default_args = {
    'owner': 'airflow',
    'start_date': days_ago(0),
    'concurrency': 1,
    'retries': 0,
    'depends_on_past': False,
}

with DAG('Annotation_Pipeline',
         catchup=False,
         default_args=default_args,
         schedule_interval='@once',
         ) as dag:
    t0_upload = PythonOperator(task_id='UploadTranscripts',
                               python_callable=push_models_to_s3)
    t1_labelling = PythonOperator(task_id='Labelling',
                                  python_callable=label_models)

t0_upload >> t1_labelling


