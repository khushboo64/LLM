from transformers import BertModel, AutoTokenizer
import pandas as pd
from scipy.spatial.distance import cosine


model_name = "bert-base-cased"

model = BertModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "Tokenize me thois please"
encoded_inputs = tokenizer(text, return_tensors='pt')
encoded_inputs
output = model(**encoded_inputs)

last_hidden_state = output.last_hidden_state
pooler_output = output.pooler_output
last_hidden_state.shape
pooler_output.shape


def predict(text):
    encoded_inputs = tokenizer(text, return_tensors='pt')
    return model(**encoded_inputs)[0]

sentence1 = "They saw a black bear."
sentence2 = "You will have to bear the pain."

token1 = tokenizer.tokenize(sentence1)
token2 = tokenizer.tokenize(sentence2)

out1 = predict(sentence1)
out2 = predict(sentence2)

emb1 = out1[0, token1.index("bear"), :].detach()
emb2 = out1[0, token2.index("bear"), :].detach()

emb1.shape
emb2.shape

cosine(emb1,emb2)

