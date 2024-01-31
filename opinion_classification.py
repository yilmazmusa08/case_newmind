import pandas as pd
from transformers import pipeline
import torch

df_topic = pd.read_csv("topics.csv")
df_opinions = pd.read_csv("opinions.csv")

# Specify the device
device = 0  # Use 0 for the first GPU, or 'cuda:0'
classifier = pipeline("zero-shot-classification",
                      model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
                      device=device)

# Print the current device
print("Current device:", torch.cuda.current_device())

sequence_to_classify = df_opinions["text"]
sequence_to_classify = list(sequence_to_classify)

candidate_labels = ['claim', 'counterclaim', 'evidence', 'rebuttal']

# Perform classification
predictions = classifier(sequence_to_classify, candidate_labels)

# Create a DataFrame
result_df = pd.DataFrame({
    'sentences': sequence_to_classify,
    'predictions': predictions
})

result_df.to_csv("opinion_pred.csv", index=False)
