import pandas as pd
from transformers import pipeline
import torch

def split_sentences(sentence, max_length=1024):
    """
    Split a long sentence into chunks of maximum specified length.
    """
    return [sentence[i:i+max_length] for i in range(0, len(sentence), max_length)]

df_topic = pd.read_csv("topic_grouping_results.csv")

# Specify the device
device = 0  # Use 0 for the first GPU, or 'cuda:0'
summarizer = pipeline("summarization",
                      model="kabita-choudhary/finetuned-bart-for-conversation-summary",
                      device=device,
                      max_length=100,  # Set your desired maximum summary length
                      min_length=20,   # Set your desired minimum summary length
                      length_penalty=2.0  # Set your desired length penalty
                      )

# Print the current device
print("Current device:", torch.cuda.current_device())

group_of_sentences = df_topic["text"].tolist()

# Perform summarization
summaries = []

for sentence_group in group_of_sentences:
    # Ensure each sentence_group has at most 1024 tokens
    sentence_chunks = split_sentences(sentence_group)
    print("Processing Sentence: ", sentence_group)
    
    # Perform summarization on the entire sentence_group
    summary = summarizer(sentence_chunks)
    
    # Append the summary to the list
    summaries.append(summary[0]['summary_text'])
    print("Added Summary: ", summary[0]['summary_text'])

# Create a DataFrame
result_df = pd.DataFrame({
    'sentences': group_of_sentences,
    'conclusion': summaries
})

result_df.to_csv("conclusion_results.csv", index=False)