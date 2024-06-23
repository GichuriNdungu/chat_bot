import pandas as pd 
import tensorflow as tf 
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam

df = pd.read_csv('data/tokenized_carbon_market.csv')

import ast

df['tokenized_question'] = df['tokenized_question'].apply(ast.literal_eval)
df['tokenized_answer'] = df['tokenized_answer'].apply(ast.literal_eval)
df['attention_mask_question'] = df['attention_mask_question'].apply(ast.literal_eval)
df['attentiomn_mask_answer'] = df['attention_mask_answer'].apply(ast.literal_eval)

questions = list(df['tokenized_question'])
answers = list(df['tokenized_answer'])
question_masks = list(df['attention_mask_question'])
answer_masks = list(df['attentiomn_mask_answer'])

def encode_dataset(questions, question_masks, answers, answer_masks):
    '''function to encode the entire dataset'''
    input_ids = tf.ragged.constant(questions)
    attention_masks = tf.ragged.constant(question_masks)
    labels = tf.ragged.constant(answers)
    labels_attention_masks = tf.ragged.constant(answer_masks)

    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': input_ids.to_tensor(default_value=0),
            'attention_mask': attention_masks.to_tensor(default_value=0)
        },
        {
            'labels': labels.to_tensor(default_value=0),
            'labels_attention_mask':labels_attention_masks.to_tensor(default_value=0)
        }
    ))
    return dataset

train_dataset = encode_dataset(questions, question_masks, answers, answer_masks)
train_dataset = train_dataset.shuffle(100).batch(8).prefetch(tf.data.experimental.AUTOTUNE)

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

optimizer = Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

model.fit(train_dataset, epochs=3)


print('compile and finetune complete')
