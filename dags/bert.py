import os
import boto3
from io import StringIO
import pandas as pd
import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
from official.nlp import optimization
from sklearn.model_selection import train_test_split

tf.get_logger().setLevel('ERROR')

# Change your access keys
ACCESS_KEY = 'access_key'
SECRET_KEY = 'aws_access_secret_key'
BUCKET = 'bucket_name'


def bert_model():
    # fetching the labelled data from s3
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)
    csv_obj = s3.get_object(Bucket=BUCKET, Key='sec-edgar/annotation/text_final.csv')
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    text_df = pd.read_csv(StringIO(csv_string))
    text_df.head()

    # Making directories so we can make the data
    os.mkdir('text_data')
    os.mkdir('text_data/train')
    os.mkdir('text_data/test')
    os.mkdir('text_data/train/pos')
    os.mkdir('text_data/train/neg')
    os.mkdir('text_data/test/pos')
    os.mkdir('text_data/test/neg')

    # split into train test sets
    train, test = train_test_split(text_df, test_size=0.2)

    for i in range(len(train)):
        if train.iloc[i]['label'] == 'positive':
            f = open('text_data/train/pos/file' + str(i) + '.txt', "w")
            f.write(train.iloc[i]['text'])
            f.close()
        elif train.iloc[i]['label'] == 'negative':
            f = open("text_data/train/neg/file" + str(i) + '.txt', "w")
            f.write(train.iloc[i]['text'])
            f.close()

    for i in range(len(test)):
        if test.iloc[i]['label'] == 'positive':
            f = open('text_data/test/pos/file' + str(i) + '.txt', "w")
            f.write(test.iloc[i]['text'])
            f.close()
        elif test.iloc[i]['label'] == 'negative':
            f = open("text_data/test/neg/file" + str(i) + '.txt', "w")
            f.write(test.iloc[i]['text'])
            f.close()

    # The dataset has already been divided into train and test, and now we are creating a validation set using an 80:20 split of the training data by using the validation_split

    AUTOTUNE = tf.data.AUTOTUNE
    batch_size = 32
    seed = 42

    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'text_data/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed)

    class_names = raw_train_ds.class_names
    train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'text_data/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed)

    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'text_data/test',
        batch_size=batch_size)

    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Selecting the bert model to run our model
    bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8'

    map_name_to_handle = {
        'small_bert/bert_en_uncased_L-4_H-512_A-8':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1', }

    map_model_to_preprocess = {
        'small_bert/bert_en_uncased_L-4_H-512_A-8': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3', }
    tfhub_handle_encoder = map_name_to_handle[bert_model_name]
    tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]
    print(f'BERT model selected : {tfhub_handle_encoder}')
    print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

    # Creating a very simple fine-tuned model, with the preprocessing model, the selected BERT model, one Dense and a Dropout layer.
    def build_classifier_model():
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
        return tf.keras.Model(text_input, net)

    # Loss function
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()

    # Optimizer
    epochs = 5
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    # Loading the BERT model and training
    classifier_model = build_classifier_model()
    classifier_model.compile(optimizer=optimizer,
                             loss=loss,
                             metrics=metrics)

    print(f'Training model with {tfhub_handle_encoder}')
    history = classifier_model.fit(x=train_ds,
                                   validation_data=val_ds,
                                   epochs=epochs)

    # Saving the model to S3
    classifier_model.save('/Users/ng/Desktop/Assignment_2/Microservices/app/bert', include_optimizer=False)

    s3_resource = boto3.resource("s3", region_name="us-east-1", aws_access_key_id=ACCESS_KEY,
                                 aws_secret_access_key=SECRET_KEY)
    try:
        # put s3 bucket name here
        bucket_name = "BUCKET"
        root_path = '/Users/ng/Desktop/Assignment_2/Microservices/app/bert/'  # local folder for upload
        my_bucket = s3_resource.Bucket(bucket_name)

        for path, subdirs, files in os.walk(root_path):
            path = path.replace("\\", "/")
            directory_name = path.replace(root_path, "")
        for file in files:
            my_bucket.upload_file(os.path.join(path, file), 'sec-edgar/model/' + directory_name + '/' + file)
    except Exception as err:
        print(err)
    s3_resource.Bucket('BUCKET').upload_file("/Users/ng/Desktop/Assignment_2/Microservices/app/bert/saved_model.pb", "sec-edgar/model/saved_model.pb")