from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(0),
    'concurrency': 1,
    'retries': 0,
    'depends_on_past': False,
}

with DAG('Inference_Pipeline', default_args=default_args, schedule_interval=None, max_active_runs=1) as dag:

    t0 = BashOperator(
        task_id='Inference',
        bash_command='python /Users/ng/Desktop/Assignment_2/Inference/main.py')

t0