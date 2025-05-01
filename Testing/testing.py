import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import logging
import sys
from sklearn.metrics import mean_absolute_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from functionality_map import functionality

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
model_path = os.path.abspath("models/iltc_finetuned_model")


def load_model(model_path):
    if not os.path.exists(model_path):
        logging.error(f"Model path '{model_path}' does not exist. Please check the path.")
        exit(1)
    model = SentenceTransformer(model_path)
    logging.info("Model loaded successfully.")
    return model


def load_test_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if {'user_input', 'user_intent', 'score'}.issubset(df.columns):
            logging.info("Test data loaded successfully.")
            return df
        else:
            raise ValueError("CSV file must contain 'user_input', 'user_intent', and 'score' columns.")
    except Exception as e:
        logging.error(f"Error loading test data: {e}")
        exit(1)


def predict_intents(model, test_data):
    keywords = list(functionality.keys())
    keyword_embeddings = model.encode(keywords, convert_to_tensor=True)
    
    predictions = []
    actual_scores = test_data['score'].tolist()
    
    for user_ip in test_data['user_input']:
        user_embedding = model.encode(user_ip, convert_to_tensor=True)
        cosine_scores = util.cos_sim(user_embedding, keyword_embeddings)
        cosine_scores_list = cosine_scores.tolist()[0]
        
        max_score = max(cosine_scores_list)
        max_score_index = cosine_scores_list.index(max_score)
        matched_keyword = keywords[max_score_index]
        matched_functionality = functionality[matched_keyword]
        users_intent = f"{matched_keyword}: {matched_functionality}"
        predictions.append((users_intent, max_score))

    return predictions, actual_scores




def calculate_mae(predictions, actual_scores):
    predicted_scores = [p[1] for p in predictions]
    mae = mean_absolute_error(actual_scores, predicted_scores)
    return mae



def main():
    model = load_model(model_path)
    test_data_path = os.path.abspath("Testing/cleaned_test_data.csv")
    test_data = load_test_data(test_data_path)
    
    predictions, actual_scores = predict_intents(model, test_data)
    mae = calculate_mae(predictions, actual_scores)
    
    logging.info(f"Mean Absolute Error (MAE): {mae:.4f}")
    logging.info(f"Model Accuracy: {100 - (mae * 100):.2f}%")
    
    test_data['predicted_score'] = [p[1] for p in predictions]
    test_data['predicted_intent'] = [p[0] for p in predictions]
    
    output_file = os.path.join(os.path.dirname(__file__), "./test_results.csv")
    test_data.to_csv(output_file, index=False)
    logging.info(f"Predictions saved to {output_file}")



if __name__ == "__main__":
    main()