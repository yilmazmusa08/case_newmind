import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn.functional as F

df_topic = pd.read_csv("topics.csv")
df_opinions = pd.read_csv("opinions.csv")

val = df_opinions["topic_id"]

# Group by 'topic_id' and aggregate 'text' values into a list
grouped_df = df_opinions.groupby('topic_id')['text'].agg(list).reset_index()

# Print the grouped sentences DataFrame
print("\nGrouped Sentences Value Counts:")
print(df_opinions["topic_id"].value_counts())


sample = df_opinions[df_opinions["topic_id"] == "DECAE402BB38"]["text"]

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Sentences we want sentence embeddings for
sentences = df_opinions["text"]
sentences = list(sentences)

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Set batch size
batch_size = 100

# Calculate number of batches
num_batches = len(sentences) // batch_size + (len(sentences) % batch_size > 0)

total_batch_length = 0
total_embedding_length = 0
all_embeddings = []

for batch_idx in range(0, len(sentences), batch_size):
    # Extract batch of sentences
    batch_sentences = sentences[batch_idx:batch_idx + batch_size]

    # Print current batch size and lengths
    print(f"Batch {batch_idx + 1}/{len(sentences)}, Current Batch Size: {len(batch_sentences)}, Current Batch Length: {sum(len(sentence.split()) for sentence in batch_sentences)}")

    # Tokenize batch
    encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings for the batch
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling (replace mean_pooling with the pooling method you want to use)
    batch_embeddings = model_output.last_hidden_state.mean(dim=1)

    # Normalize embeddings
    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

    # Append batch embeddings to the overall list
    all_embeddings.append(batch_embeddings)

    # Update lengths
    total_batch_length += len(batch_sentences)
    total_embedding_length += batch_embeddings.shape[0]
    print("Total Batch Length:", total_batch_length)
    print("Total Embedding Length:", total_embedding_length)
    print("Embeddings shape:", len(all_embeddings))

# Concatenate embeddings along the first dimension
all_embeddings = torch.cat(all_embeddings, dim=0)

cosine_sim_matrix = util.pytorch_cos_sim(all_embeddings, all_embeddings)
print("cosine_sim_matrix:", cosine_sim_matrix.shape)
print("sentences:", len(sentences))

# Convert similarity matrix to a DataFrame for better readability
cosine_sim_df = pd.DataFrame(cosine_sim_matrix.numpy(), columns=sentences, index=sentences)

cosine_sim_df.to_csv("cos_sim_dataframe.csv", index=False)

# Group sentences based on similarity scores
threshold = 0.8
groups = []

# Identify pairs with values greater than the threshold
upper_diagonal_pairs = np.where(np.triu(cosine_sim_df.values, k=1) > threshold)

# Create a dictionary to keep track of which values are already used
used_values = {col: False for col in cosine_sim_df.columns}

# Sort pairs by descending order of values
sorted_pairs = sorted(zip(upper_diagonal_pairs[0], upper_diagonal_pairs[1], cosine_sim_df.values[upper_diagonal_pairs]), key=lambda x: x[2], reverse=True)

# Create groups based on sorted pairs
for i, j, _ in sorted_pairs:
    col_i, col_j = cosine_sim_df.columns[i], cosine_sim_df.columns[j]

    # Check if both values are available
    if not used_values[col_i] and not used_values[col_j]:
        group = [col_i, col_j]
        groups.append(group)

        # Mark the values as used
        used_values[col_i] = True
        used_values[col_j] = True
    elif not used_values[col_i]:
        # Add col_i to an existing group
        for group in groups:
            if col_i in group:
                group.append(col_i)
                used_values[col_i] = True
                pass
        else:
            # print(f"Marked {col_j} as used")
            group.append(col_j)
            used_values[col_j] = True

    elif not used_values[col_j]:
        # Add col_j to an existing group
        for group in groups:
            if col_j in group:
                group.append(col_j)
                used_values[col_j] = True
                pass
        else:
            # print(f"Marked {col_j} as used")
            group.append(col_j)
            used_values[col_j] = True

# Create a DataFrame for the grouped sentences using pd.concat
dfs = [pd.DataFrame({"topic_id": [f"Group {i + 1}"],
                     "text": [group]}) for i, group in enumerate(groups)]

grouped_sentences_df = pd.concat(dfs, ignore_index=True)

grouped_sentences_df.to_csv("topic_pred.csv", index=False)

