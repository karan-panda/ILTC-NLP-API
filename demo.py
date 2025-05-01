import os
import numpy as np
from sentence_transformers import SentenceTransformer, util
from functionality_map import functionality

model_path = './models/iltc_finetuned_model'

top_n = 10

if not os.path.exists(model_path):
    print(f"Model path '{model_path}' does not exist. Please check the path.")
    exit(1)

model = SentenceTransformer(model_path)
print(model)

user_ip = input("Enter your query: ")

keywords = list(functionality.keys())

user_embedding = model.encode(user_ip, convert_to_tensor=True)
keyword_embeddings = model.encode(keywords, convert_to_tensor=True)

cosine_scores = util.cos_sim(user_embedding, keyword_embeddings)
cosine_scores_list = cosine_scores.tolist()[0]

top_indices = np.argsort(cosine_scores_list)[::-1][:top_n]

top_intents = [(keywords[i], functionality[keywords[i]], cosine_scores_list[i]) for i in top_indices]

print("Top Intent Matches:")
for keyword, functionality_desc, score in top_intents:
    print(f"{keyword}: {functionality_desc} (Score: {score:.4f})")
