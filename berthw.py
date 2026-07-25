from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

tokenized_text = tokenizer.tokenize("the cat is on the mat")
print(tokenized_text)

sentences = ["This place is wonderful!", "This place is amazing!", "This place is awful!", "I love it here!"]
L = len(sentences)
hidden_states = []
norms = []
for i in range(L):
    input_ids = torch.tensor(tokenizer.encode(sentences[i])).unsqueeze(0) 
    outputs = model(input_ids)
    last_hidden_states = outputs[0]
    hidden_states.append(last_hidden_states)

for i in range(L):
    ls = []
    for j in range(L):
        ls.append(torch.norm(hidden_states[i] - hidden_states[j]).item())
    norms.append(ls)  

print(norms)


