import pandas as pd 
import tensorflow as tf 
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam

df = pd.read_csv('data/tokenized_carbon_mak')