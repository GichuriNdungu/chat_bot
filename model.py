#!/usr/bin/env python3
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split


df = pd.read_csv('data/tokenized_carbon_market.csv')
label_map = {answer: idx for idx, answer in enumerate(df['Answer'].unique())}
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
df['label'] = df['Answer'].map(label_map)

def prepare_data(df):
    input_ids = list(df['tokenized_question'].apply(eval))
    attention_masks = list(df['attention_mask_question'].apply(eval))
    labels = list(df['label'])  # Use the mapped labels
    
    return tf.data.Dataset.from_tensor_slices(({
        'input_ids': tf.constant(input_ids, dtype=tf.int32),
        'attention_mask': tf.constant(attention_masks, dtype=tf.int32)
    }, tf.constant(labels, dtype=tf.int32)))

train_df, test_df = train_test_split(df, test_size=0.2)
train_dataset = prepare_data(train_df).shuffle(100).batch(8).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = prepare_data(test_df).batch(8).prefetch(tf.data.experimental.AUTOTUNE)

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_map))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=20)
