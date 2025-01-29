import pickle

# Path to the pickle file
pickle_file_path = '/Users/juliagontijolopes/Desktop/TP1/Gensim/models_results.pkl'

with open(pickle_file_path, 'rb') as file:
    results = pickle.load(file)

print(results)