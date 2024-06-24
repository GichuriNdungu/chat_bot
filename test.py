import json
with open('data/preprocessed_dataset.json', 'r') as f:
    data = json.load(f)

for article in data['data']:
    for paragraph in article['paragraphs']:
        for qa in paragraph['qas']:
            print(qa['question'])