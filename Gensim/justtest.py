from gensim.models import Word2Vec
from gensim.models.word2vec import Text8Corpus

from gensim.test.utils import datapath
from gensim import utils
from tqdm import tqdm

import os
import requests
import zipfile

import gensim.models
import itertools
import numpy as np

from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

output_folder = "models"
os.makedirs(output_folder, exist_ok=True)
current_directory = os.getcwd()
test_file_path = os.path.join(current_directory, "data", "questions-words.txt")
text8_path = os.path.join(current_directory, "data", "text8")


## MODEL TRAINING AND EVALUATION
def get_model(corpus, hyperparameters):
    model_type, window_size, vector_size, steps = hyperparameters
    sg = 0 if model_type == 'CBOW' else 1
    model_name = f"{model_type}_w{window_size}_v{vector_size}_s{steps}.model"

    try:
        model = Word2Vec.load(f"{output_folder}/{model_name}")
        return model, model_name
    except Exception as e:
        print(f"\nError training {model_type} with window={window_size}, vector_size={vector_size}, steps={steps}: {e}")
        return None, model_name


# Create, train and save each model
def train_and_evaluate_models(corpus_path, test_data_path, combinations):
    corpus = Text8Corpus(corpus_path)
    test_data = get_test_data(test_data_path) 

    trained_models = []
    failed_models = []

    models_evaluation_results = {}

    # tqdm for tracking combinations
    for comb in tqdm(combinations, desc="Treinando e avaliando modelos"):
        # Train model
        model, model_name = get_model(corpus, comb)
        if model:
            trained_models.append(model_name)
            
            # Evaluate model
            model_mean_distance, model_mean_similarity, model_accuracy = test_model(model, test_data)
            models_evaluation_results[model_name] = {
                "mean_distance" : model_mean_distance,
                "mean_similarity" : model_mean_similarity,
                "accuracy" : model_accuracy
            }

        else: 
            failed_models.append(model_name)
    
    return trained_models, failed_models, models_evaluation_results


def get_test_data(test_file_path):
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"Test file '{test_file_path}' not found.")
    
    with open(test_file_path, "r") as f:
        lines = f.readlines()
    
    words = [line.strip().lower().split() for line in lines if line.strip() and not line.startswith(":")]
    return words


# Test each model with analogies
def test_model(model, test_data):
    cosine_distances = []
    similarities = []
    correct_predictions = 0
    total_predictions = 0
    missing_words = set()

    for test_sample in test_data:
        if len(test_sample) != 4:
            # Skip malformed test samples
            continue
        a, b, c, expected = test_sample
        if all(word in model.wv for word in [a, b, c, expected]):
            try:
                # Perform the analogy task: a is to b as c is to ?
                analogy_result, _ = model.wv.most_similar(positive=[c, b], negative=[a], topn=1)[0]
                predicted = model.wv[c] + model.wv[b] - model.wv[a]
                
                # Calculate similarity between predicted and expected
                cosine_distance = cosine(predicted, model.wv[expected])
                similarity = cosine_similarity(
                    [predicted],
                    [model.wv[expected]]
                )[0][0]

                similarities.append(similarity)
                cosine_distances.append(cosine_distance)

                # Update accuracy
                if analogy_result == expected:
                    correct_predictions += 1
                total_predictions += 1
            except KeyError as e:
                print(f"Error with words {test_sample}: {e}")
        else:
            missing_words.update(word for word in [a, b, c, expected] if word not in model.wv)
            # print(f"Skipping test sample {test_sample} due to missing words: {missing}")

    print(f"{len(missing_words)} missing words = {missing_words}\n")
    # Compute metrics
    if total_predictions > 0:
        average_distance = sum(cosine_distances) / len(cosine_distances)
        average_similarity = sum(similarities) / len(similarities)
        accuracy = correct_predictions / total_predictions
        return average_distance, average_similarity, accuracy
    else:
        return None, None, None


import pickle
def save_evaluation_results(results, file_path):
    # Save as a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to {file_path} in pickle format.")




model_types = ['CBOW', 'Skip-gram']
vector_sizes = [50, 100, 200]
window_sizes = [2, 5, 10]
num_steps = [5, 10, 20]

combinations = list(itertools.product(model_types, window_sizes, vector_sizes, num_steps))
print(combinations)

print(f"Number of models: {len(combinations)}")

trained_models, failed_models, models_evaluation_results = train_and_evaluate_models(text8_path, test_file_path, combinations)
save_evaluation_results(models_evaluation_results, "models_results_final.pkl")

if failed_models:
    file_path = "failed_models.txt"
    with open(file_path, "a") as f:
        for model in failed_models:
            f.write(model + "\n")
    
    print(f"Appended {len(failed_models)} entries to {file_path}.")