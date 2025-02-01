# Plagiarism-Detection
NLP project
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to calculate the similarity between documents
def calculate_similarity(documents):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity_matrix

# Example documents (you can replace these with your own documents)
documents = [
    "This is a sample document.",
    "This document is a sample.",
    "Completely different content."
]

# Calculate similarity
similarity_matrix = calculate_similarity(documents)

# Display results
print("Similarity Matrix:")
print(similarity_matrix)
