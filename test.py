from transformers import DebertaTokenizer, DebertaModel

tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
model = DebertaModel.from_pretrained("microsoft/deberta-base")

# inputs = tokenizer(, return_tensors="pt")
inputs = tokenizer.batch_encode_plus(["Hello, my dog is cute"],
                                     padding='max_length',
                                     truncation=True,
                                     return_tensors="pt")
# print(inputs)
# inputs:  { input_ids  'token_type_ids  attention_mask }
outputs = model(**inputs)
print(outputs.last_hidden_state.shape) 
#outputs：{ last_hidden_state grad_fn=<AddBackward0>) hidden_states=None  attentions=None }
# last_hidden_state.shape  [1, 512, 768] batchSize sentenceSize hiddenSize

# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states)