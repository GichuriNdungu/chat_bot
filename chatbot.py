from transformers import BertTokenizer, TFBertForQuestionAnswering
import tensorflow as tf
import requests

# Load the saved model and tokenizer
model_load_path = 'models/bert-base-uncased'
tokenizer_load_path = 'models/tokenizers'

tokenizer = BertTokenizer.from_pretrained(tokenizer_load_path)
model = TFBertForQuestionAnswering.from_pretrained(model_load_path)

def fetch_context_from_wikipedia(question):
    """Fetches a context paragraph from Wikipedia based on the question."""
    session = requests.Session()
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": question
    }
    response = session.get(url=url, params=params)
    data = response.json()
    page_title = data['query']['search'][0]['title']
    
    # Fetch the extract of the page
    params = {
        "action": "query",
        "format": "json",
        "titles": page_title,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
    }
    response = session.get(url=url, params=params)
    page_id = next(iter(response.json()['query']['pages']))
    extract = response.json()['query']['pages'][page_id]['extract']
    return extract
def chat():
    print("Chatbot is running (type 'quit' to stop)...")
    while True:
        user_input = input("You (ask a question): ")
        if user_input.lower() == 'quit':
            break

        context = fetch_context_from_wikipedia(user_input)
        print(context[:500])
        inputs = tokenizer.encode_plus(user_input, context, return_tensors="tf", padding=True, truncation=True, max_length=512)
        answer_start_scores, answer_end_scores = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits

        answer_start = tf.argmax(answer_start_scores, axis=1).numpy()[0]
        answer_end = tf.argmax(answer_end_scores, axis=1).numpy()[0] + 1

        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0, answer_start:answer_end]))
        print(f"Chatbot: {answer}")

if __name__ == "__main__":
    chat()