# Imports the modules
import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

# Import Custom Modules
import bert


def ml_pipeline():
    bert.bert_model()


default_args = {
    'owner': 'airflow',
    'start_date': days_ago(0),
    'concurrency': 1,
    'retries': 0,
    'depends_on_past': False,
}

with DAG('ML_Pipeline',
         catchup=False,
         default_args=default_args,
         schedule_interval='@once',
         ) as dag:
    t0_ml_pipeline = PythonOperator(task_id='Model',
                                    python_callable=ml_pipeline)

t0_ml_pipeline 
