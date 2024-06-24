import json
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Initialize the summarizer
summarizer = pipeline('summarization')

def summarize_text(text, max_length=50):
    summarized = summarizer(text, max_length=max_length, min_length=25, do_sample=False)
    return summarized[0]['summary_text']

def reformulate_question(section_text):
    return f"What is {section_text.strip()}?"

def scrape_wikipedia_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    q_and_a = []
    for section in soup.find_all('h2'):
        question = reformulate_question(section.text)
        answer_section = section.find_next_sibling('p')
        if answer_section:
            answer = answer_section.text.strip()
            summarized_answer = summarize_text(answer)
            q_and_a.append((question, summarized_answer))
        else:
            print(f'No answer found for question: {question}')
    return q_and_a

urls = [
    'https://en.wikipedia.org/wiki/Carbon_emission_trading',
    'https://en.wikipedia.org/wiki/Carbon_offsets_and_credits',
    'https://en.wikipedia.org/wiki/Emissions_trading'
]

all_data = []
for url in urls:
    title = url.split('/')[-1].replace('_', ' ')
    q_and_a_pairs = scrape_wikipedia_page(url)
    paragraphs = []

    for question, answer in q_and_a_pairs:
        paragraphs.append({
            'context': answer,
            'qas': [{
                'question': question,
                'id': f'{title}_{len(paragraphs)}',
                'answers': [{
                    'text': answer,
                    'answer_start': 0
                }]
            }]
        })

    all_data.append({
        'title': title,
        'paragraphs': paragraphs
    })

squad_format = {'data': all_data, 'version': '1.0'}

# Save the data in SQuAD format
with open('data/carbon_market_qa.json', 'w') as json_file:
    json.dump(squad_format, json_file, indent=2)

print('Data saved in SQuAD format.')
