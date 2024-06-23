import pandas as pd 
from transformers import BertTokenizer

df = pd.read_csv('data/carbon_market_qa.csv')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_text(text):
    return tokenizer.encode(text, add_special_tokens = True)
df['tokenized_question'] = df['Question'].apply(tokenize_text)
df['tokenized_answer'] = df['Answer'].apply(tokenize_text)

#create tensors and attention masks for tokenized data

def create_attention_mask(tokens):
    return[1] * len(tokens)

df['attention_mask_question'] = df['tokenized_question'].apply(create_attention_mask)
df['attention_mask_answer'] = df['tokenized_answer'].apply(create_attention_mask)

# save tokenized to new file 

df.to_csv('data/tokenized_carbon_market.csv', index=False)

print('Tokenization completed')
