import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr
from plot_utils import plot_embeddings, plot_combined_embeddings, compute_rsa

# Simulate loading fMRI data
def get_fmri_data():
    # Replace with actual loading code
    words = ["dog", "cat", "lion", "tiger", "elephant", "hammer", "screwdriver", "wrench", "drill", "saw", "plunger"]
    np.random.seed(0)
    fmri_vectors = np.random.rand(len(words), 325325)  # Dummy fMRI data
    return words, fmri_vectors

# Get BERT embeddings
def get_bert_embeddings(words):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(words)
    print(f"[DEBUG] Got {len(embeddings)} BERT vectors for {len(words)} words")
    return embeddings

if __name__ == "__main__":
    print("Getting BERT embeddings...")
    words = ["dog", "cat", "lion", "tiger", "elephant", "hammer", "screwdriver", "wrench", "drill", "saw", "plunger"]
    bert_vectors = get_bert_embeddings(words)

    print("Getting fMRI data...")
    fmri_words, fmri_vectors = get_fmri_data()

    # Sanity check and alignment
    min_len = min(len(bert_vectors), len(fmri_vectors), len(words))
    bert_vectors = bert_vectors[:min_len]
    fmri_vectors = fmri_vectors[:min_len]
    words = words[:min_len]  # Using the original word list you used for BERT

    print("Plotting individual PCAs...")
    plot_embeddings(bert_vectors, words, "BERT Embeddings (PCA)")
    plot_embeddings(fmri_vectors, words, "fMRI Activations (PCA)")

    print("Plotting combined PCA...")
    plot_combined_embeddings(bert_vectors, fmri_vectors, words)

    print("Computing Representational Similarity (RSA)...")
    score = compute_rsa(bert_vectors, fmri_vectors)
    print(f"RSA (Spearman correlation between similarity matrices): {score:.4f}")
