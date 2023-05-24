import pandas as pd
import numpy as np
import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer, AutoTokenizer
# from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
import torch.nn.functional as F
from collections import Counter

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Installations
# !pip install transformers
# !pip install -U torch torchvision
# !sudo apt-get install ninja-build


DATA  = "NN_improve_tweets_new.csv"
df = pd.read_csv(DATA)

"""## Data Preprocessing"""
df = df[['text','hate']]

# Define the label mapping
label_map = {
    'normal': 0,
    'offensive': 1,
    'hateful': 2
}

output_list = []
for i in df.index:
  if df.loc[i, 'hate'] == 1 or  df.loc[i, 'hate'] == 2:
    output_list.append([df.loc[i, 'text'], 1])
  else:
    output_list.append([df.loc[i, 'text'], 0])

output_df = pd.DataFrame(output_list, columns=['text', 'hate'])
print(output_df)
df = output_df

# Split the data into training and validation sets
train_df, dev_df, test_df =  np.split(df.sample(frac=1, random_state=42),[int(.6*len(df)), int(.8*len(df))])
print(train_df.shape, dev_df.shape, test_df.shape)


# check the distribution of number of tokens of the dataset
# def check_maxlen(dataframe):
#   inputs = {
#           "input_ids":[],
#           "attention_mask":[]
#         }

#   sents = dataframe['tweet'].values.tolist()
#   lengths = []
#   for sent in sents:
#     tokenized_input = tokenizer(sent)
#     lengths.append(len(tokenized_input["input_ids"])) 
  
#   return lengths

# lengths = check_maxlen(df)
# plt.hist(lengths)



"""## Prepare Model"""
# print(torch.__version__)
# print(torch.version.cuda)
# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#load model
num_labels = 2 # binary classification
model = BertForSequenceClassification.from_pretrained('vinai/bertweet-base',num_labels = num_labels)

# Tell pytorch to run this model on the GPU.
desc = model.cuda()

tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', use_fast=True)

"""## Appling Model on Dataset to extract high probability tweets"""

#create custom dataset 
class TweetDataset(Dataset):
    def __init__(self, encodings, labels):
      self.encodings = encodings
      self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['label'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

MAX_LENGTH = 128
def create_dataset(dataframe):
  inputs = {
          "input_ids":[],
          "attention_mask":[]
        }

  sents = dataframe['tweet'].values.tolist()
  for sent in sents:
    tokenized_input = tokenizer(sent,max_length=MAX_LENGTH, padding='max_length', truncation = True)
    inputs["input_ids"].append(torch.tensor(tokenized_input["input_ids"]))
    inputs["attention_mask"].append(torch.tensor(tokenized_input["attention_mask"]))

  # Create a TensorDataset from the input data
  labels = torch.tensor([0]*dataframe.shape[0])
  return TweetDataset(inputs, labels)

train_dataset = create_dataset(train_df)
dev_dataset = create_dataset(dev_df)
test_dataset = create_dataset(test_df)
print(test_dataset)



# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='steps',
    load_best_model_at_end=True
)

num_labels = 2 # binary classification
model = BertForSequenceClassification.from_pretrained('vinai/bertweet-base',num_labels = num_labels)

# Tell pytorch to run this model on the GPU.
# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
)

# Fine-tune the model
trainer.train()

# Load the model and tokenizer
num_labels = 2 # binary classification
path = "/content/drive/MyDrive/my_models/improved_bertweet_model"
model = BertForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels = num_labels)

# Save the trained model
model.save_pretrained(path)

# Evaluate the model on the test dataset
results = trainer.evaluate(test_dataset)
print(results)
# print(f"Precision: {results['eval_precision']}")
# print(f"Recall: {results['eval_recall']}")


num_labels = 2 # binary classification
path = "/content/drive/MyDrive/my_models/improved_bertweet_model"
model = BertForSequenceClassification.from_pretrained(path)
# Tell pytorch to run this model on the GPU.
# desc = model.cuda()

# Compute predictions using Trainer
test_args = TrainingArguments(
    output_dir = "./result_2e_5",
    do_train = False,
    do_predict = True,
    per_device_eval_batch_size = 16,   
)

trainer = Trainer(model=model, args =test_args)
output = trainer.predict(test_dataset)

probabilities = F.softmax(torch.from_numpy(output.predictions), dim=-1)

# sort by high probability
high_prob = torch.max(probabilities, dim = 1)
print(high_prob)

sorted, index = high_prob.values.sort(descending=True)
print(sorted, index) # we know the index of the tweets that have high prob

