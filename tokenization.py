import json
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

dataset_file = 'data/preprocessed_dataset.json'

with open(dataset_file, 'r', encoding='utf-8') as f:
    dataset = json.load(f)


paragraphs = []
questions = []

for data in dataset['data']:
    for paragraph in data['paragraphs']:
        paragraphs.append(paragraph['context'])
        for qa in paragraph['qas']:
            questions.append(qa['question'])

#split
train_paragraphs, val_paragraphs = train_test_split(paragraphs, test_size=0.3, random_state=42)
train_questions, val_questions = train_test_split(questions, test_size=0.3, random_state=42)

print(train_questions)

#start tokenization

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokenized_train_questions = tokenizer(train_questions, padding=True, truncation=True, return_tensors="tf")
tokenized_val_questions = tokenizer(val_questions, padding=True, truncation=True, return_tensors="tf")


