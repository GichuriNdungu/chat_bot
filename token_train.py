import json
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFBertForQuestionAnswering
import tensorflow as tf
from datasets import Dataset, DatasetDict

# Load the dataset
dataset_file = 'data/preprocessed_dataset.json'
from datasets import load_dataset

dataset = load_dataset("rajpurkar/squad")
from datasets import load_dataset

train_split = dataset["train"].train_test_split(test_size=0.02)

validation_split = dataset["validation"].train_test_split(test_size=0.02)

smaller_dataset = DatasetDict({
    "train": train_split["test"],  # 10% of the original training dataset
    "validation": validation_split["test"]  # 10% of the original validation dataset
})

print(smaller_dataset)
# with open(dataset_file, 'r', encoding='utf-8') as f:
#     dataset = json.load(f)

# def createdataset(_data):
#     contexts = []
#     questions = []
#     answers = []

#     for i in _data['data']:
#         for j in i['paragraphs']:
#             context = j['context']
#             for k in j['qas']:
#                 question = k['question']
#                 for m in k['answers']:
#                     contexts.append(context)
#                     questions.append(question)
#                     answers.append({'text': m['text'], 'answer_start': m['answer_start']})
    
#     return Dataset.from_dict({
#     'context': contexts,
#     'question': questions,
#     'answers': answers
#     })
# train = createdataset(squad)

# train = train.train_test_split(test_size=0.2, seed=4444)

# print(train)

#start tokenization

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def tokenize_function(examples):
    inputs = tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length = 512)
    start_positions = []
    end_positions = []
    for i,answer in enumerate(examples['answers']):
        start_position = answer['answer_start'][0]
        end_position = start_position + len(answer['text'][0])
        start_positions.append(inputs.char_to_token(i, start_position))
        end_positions.append(inputs.char_to_token(i, end_position -1))
        if start_positions[-1] is None or end_positions[-1] is None:
            start_positions[-1] = tokenizer.cls_token_id
            end_positions[-1] = tokenizer.cls_token_id
    inputs.update({'start_positions': start_positions, 'end_positions': end_positions})
    return inputs

tokenized_train = smaller_dataset.map(tokenize_function, batched=True)

print(tokenized_train)

# print(tokenized_train['train']['start_positions'])

from transformers import TFAutoModelForQuestionAnswering, TrainingArguments, Trainer

model = TFAutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

train_dataset = tokenized_train['train']
train_dataset = tf.data.Dataset.from_tensor_slices((
    {
        'input_ids': train_dataset['input_ids'],
        'attention_mask': train_dataset['attention_mask'],
        'token_type_ids': train_dataset['token_type_ids']
    },
    {
        'start_positions': train_dataset['start_positions'],
        'end_positions': train_dataset['end_positions']
    }
))

print(train_dataset)
train_dataset = train_dataset.shuffle(1000).batch(4)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
              loss={'start_positions': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    'end_positions': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)},
              metrics=['accuracy'])

model.fit(train_dataset, epochs=1)
