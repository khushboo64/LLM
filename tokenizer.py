from transformers import BertModel, AutoTokenizer
import pandas as pd

model_name = "bert-base-cased"

model = BertModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

sentence = "When life gives you lemons, don't make lemonade."

tokens = tokenizer.tokenize(sentence)

tokens

vocab = tokenizer.vocab
vocab_df = pd.DataFrame({"token":vocab.keys, "token_id":vocab.values()})
vocab_df = vocab_df.sort_values(by='token_id').set_index('token_id')
vocab_df

token_ids = tokenizer.encode(sentence)
token_ids

list(zip(tokens, token_ids[1:-1]))

tokenizer.decode(token_ids)
tokenizer.decode(token_ids[1:-1])


tokenizer_out = tokenizer(sentence)
tokenizer_out
sentence2 = sentence.replace("don't", "")
tokenizer_out2 = tokenizer([sentence, sentence2], padding=True)
tokenizer_out2

tokenizer.decode(tokenizer_out2['input_ids'][1])
tokenizer.decode(tokenizer_out2['input_ids'][0])

