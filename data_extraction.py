import pandas as pd
import requests
from bs4 import BeautifulSoup

# Example function to scrape Q&A from Wikipedia
def scrape_wikipedia_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    q_and_a = []
    
    # Example: Extracting Q&A pairs from Wikipedia's Carbon Market page
    for section in soup.find_all('h2'):
        question = section.text
        print(question)
        answer_section = section.find_next('p')
        if answer_section:
            answer = answer_section.text
            q_and_a.append((question, answer))
        else:
            print(f'No answer found for question: {question}')
    
    return q_and_a

# URL of Wikipedia page on carbon markets
urls = ['https://en.wikipedia.org/wiki/Carbon_emission_trading',
        'https://en.wikipedia.org/wiki/Carbon_offsets_and_credits',
        'https://en.wikipedia.org/wiki/Emissions_trading']

all_q_and_a_pairs = []
for url in urls:
    q_and_a_pairs = scrape_wikipedia_page(url)
    all_q_and_a_pairs.extend(q_and_a_pairs)

# Convert to DataFrame
df = pd.DataFrame(all_q_and_a_pairs, columns=['Question', 'Answer'])

# Clean and preprocess data
df['Question'] = df['Question'].str.lower().str.replace(r'[^a-zA-Z0-9\s]', '')
df['Answer'] = df['Answer'].str.lower().str.replace(r'[^a-zA-Z0-9\s]', '')

# Save to CSV
df.to_csv('data/carbon_market_qa.csv', index=False)
