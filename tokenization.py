import pandas as pd 
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
df = pd.read_csv('data/carbon_market_qa.csv')

train_texts, test_texts, train_labels, test_labels = train_test_split(df['Question'].tolist(), df['Answer'].tolist(), test_size=0.2)

label_map = {label: idx for idx, label in enumerate(df['Answer'].unique())}
df['label'] = df['Answer'].map(label_map)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize_text(text):
    return tokenizer.encode(text, add_special_tokens = True, max_length = 128, truncation=True, padding='max_length')
df['tokenized_question'] = df['Question'].apply(tokenize_text)
# df['tokenized_answer'] = df['Answer'].apply(tokenize_text)

#create tensors and attention masks for tokenized data

def create_attention_mask(tokens):
    return[1] * len(tokens)

df['attention_mask_question'] = df['tokenized_question'].apply(create_attention_mask)
# df['attention_mask_answer'] = df['tokenized_answer'].apply(create_attention_mask)


df[['tokenized_question', 'attention_mask_question', 'Answer']].to_csv('data/tokenized_carbon_market.csv', index=False)

print('Tokenization completed')
