# Question Answering System Chatbot

This project is an end to end Question Ansering chatbot designed to answer questions specific to the carbon markets, posed in natural language. 
It leverages a pretrained deep learning model from BERT, specifically the BERT Question answering model. 
## Data extraction

This project leverages a customized data set developed through web scraping wikipedia articles on the subject matter

The script `data_extraction.py`, is designed to extract information related to carbon emmissions trading from wikipedia.
H2 tags from wikipedia are reformated into questions, and the corresponding paragraphs summarized into answers.
lastly, this script formats and saves all the data into the Stanford Question Ansering Dataset (SQUAD) format.

## Data Preprocessing

The script `preprocess.py` preprocesses the extracted data further by lowering uppercase letters. 
Further, the script leverages regular expressions to remove all non-word characters, effectively stripping punctuation from the text.
Summatively, the script builds a new data structure that mirros the scripted data, but with preprocessed text. This structure includes titles, each associated with preprocessed paragraphs(preprocessed context and question-answering pairs)
finally, the script returns a dictionary containing preprocessed data under the key 'data' and the extracted data under the key 'version'.

## Model training and saving. 

This script `model.py` provides an end-to-end solution for training a question-answering model using the BERT architecture. It covers data loading, preprocessing, tokenization, model training, and saving, making it a comprehensive tool for developers working on question-answering systems.

The script performs the following major tasks:

1. **Data Loading**: Loads a preprocessed dataset from a JSON file.
2. **Dataset Creation**: Processes the loaded data to create a dataset suitable for training a question-answering model.
3. **Data Tokenization**: Uses a tokenizer to convert text data into a format that can be fed into the BERT model.
4. **Model Training**: Trains a BERT model on the tokenized dataset.
5. **Evaluation Metric Definition**: Defines a custom F1 score metric for model evaluation.
6. **Model Saving**: Saves the trained model and tokenizer for future use.

### Usage

This script is intended to be run as a standalone Python program. Ensure that the required libraries (`numpy`, `sklearn`, `tensorflow`, `transformers`, `datasets`) are installed in your environment. The script expects a preprocessed dataset in JSON format located at `data/preprocessed_dataset.json`.

## CLI chatbot.

The script `chatbot.py` implements a chatbot that answers questions by drawing from the pretrained BERT model.

The chatbot operates by performing the following steps:
1. **Model and Tokenizer Loading**: Loads a pre-trained BERT model and tokenizer.
2. **Wikipedia Context Fetching**: Fetches a context paragraph from Wikipedia based on the user's question.
3. **Question Answering**: Uses the BERT model to find the answer within the fetched context.
4. **Interaction Loop**: Continuously accepts user questions and provides answers until the user decides to quit.

### Usage

To use this chatbot, ensure you have the required libraries installed (`transformers`, `tensorflow`, `requests`). The script expects the BERT model and tokenizer to be pre-trained and saved at specified paths. Run the script in a Python environment, and you can start asking questions right away.

### Interaction Loop

- The script continuously prompts for questions and provides answers in a loop.
- The loop can be exited by typing 'quit'.

## Future Contributions

Future enhancements could include expanding the dataset to cover a broader range of environmental topics, improving the model's accuracy with more advanced NLP techniques, and deploying the chatbot in a web or mobile application for wider accessibility. The foundation laid by this project opens numerous avenues for exploration and innovation in leveraging AI for environmental education and awareness.