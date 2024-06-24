import json
import re

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove special characters except alphanumeric and whitespace
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_dataset(dataset):
    preprocessed_data = []
    
    for entry in dataset['data']:
        title = entry['title']
        paragraphs = entry['paragraphs']
        
        preprocessed_paragraphs = []
        
        for paragraph in paragraphs:
            context = paragraph['context']
            qas = paragraph['qas']
            
            preprocessed_qas = []
            
            for qa in qas:
                question = preprocess_text(qa['question'])
                answers = qa['answers']
                
                preprocessed_answers = []
                
                for answer in answers:
                    answer_text = preprocess_text(answer['text'])
                    answer_start = answer['answer_start']
                    
                    preprocessed_answers.append({
                        'text': answer_text,
                        'answer_start': answer_start
                    })
                
                preprocessed_qas.append({
                    'question': question,
                    'id': qa['id'],
                    'answers': preprocessed_answers
                })
            
            preprocessed_paragraphs.append({
                'context': preprocess_text(context),
                'qas': preprocessed_qas
            })
        
        preprocessed_data.append({
            'title': title,
            'paragraphs': preprocessed_paragraphs
        })
    
    return {
        'data': preprocessed_data,
        'version': dataset['version']
    }

# Load your JSON dataset
with open('data/carbon_market_qa.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# Preprocess the dataset
preprocessed_dataset = preprocess_dataset(dataset)

# Save the preprocessed dataset to a new JSON file
output_file = 'data/preprocessed_dataset.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(preprocessed_dataset, f, ensure_ascii=False, indent=2)

print(f'Preprocessed dataset saved to {output_file}')