from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import os
import time

# Get the directory of the current file
current_directory = os.path.dirname(os.path.abspath(__file__))

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for

sentences = ['This is an example sentence', 'Each sentence is converted']
# Define the file path
file_path = os.path.join(current_directory, 'sentences.txt')

# Open the file and read lines into a list
with open(file_path, 'r') as file:
    lines = file.readlines()

# Strip newline characters from each line
lines = [line.strip() for line in lines]
sentences = lines[:1000]

device = 'cuda:1'

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

model.to(device)

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
encoded_input.to(device)

# Compute token embeddings
beg = time.time()
with torch.no_grad():
    model_output = model(**encoded_input)

end = time.time()
print(f"{end - beg}")

# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

print("Sentence embeddings:")
print(sentence_embeddings)