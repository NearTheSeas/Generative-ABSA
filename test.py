
tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
model = DebertaModel.from_pretrained("microsoft/deberta-base")

# inputs = tokenizer(, return_tensors="pt")
inputs = tokenizer.batch_encode_plus(["Hello, my dog is cute"],
                                     padding='max_length',
                                     truncation=True,
                                     return_tensors="pt")
print(inputs)
# input_ids  'token_type_ids  attention_mask
# outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states)