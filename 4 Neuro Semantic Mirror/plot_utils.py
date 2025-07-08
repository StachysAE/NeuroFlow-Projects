import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import seaborn as sns
import numpy as np

animal_words = {"dog", "cat", "lion", "tiger", "elephant"}

def plot_embeddings(vectors, words, title):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)

    # Defensive check to avoid index errors
    if len(reduced) != len(words):
        print(f"[WARNING] Length mismatch: {len(reduced)} PCA vectors vs {len(words)} words")
        min_len = min(len(reduced), len(words))
        reduced = reduced[:min_len]
        words = words[:min_len]

    plt.figure(figsize=(8, 6))
    for i, word in enumerate(words):
        plt.scatter(reduced[i, 0], reduced[i, 1], color='blue')
        plt.text(reduced[i, 0], reduced[i, 1], word, fontsize=9, color='blue')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_combined_embeddings(bert_vectors, fmri_vectors, words):
    # PCA separately for both
    pca_bert = PCA(n_components=2)
    bert_2d = pca_bert.fit_transform(bert_vectors)

    pca_fmri = PCA(n_components=2)
    fmri_2d = pca_fmri.fit_transform(fmri_vectors)

    plt.figure(figsize=(12, 8))
    plt.scatter(bert_2d[:, 0], bert_2d[:, 1], color='blue', label='BERT')
    plt.scatter(fmri_2d[:, 0], fmri_2d[:, 1], color='red', marker='x', label='fMRI')

    for i, word in enumerate(words):
        plt.text(bert_2d[i, 0], bert_2d[i, 1], word, fontsize=8, color='blue')
        plt.text(fmri_2d[i, 0], fmri_2d[i, 1], word, fontsize=8, color='red')

    plt.title("Word Representations: BERT vs fMRI (PCA Reduced)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_rsa(emb1, emb2):
    # Compute similarity matrices
    sim1 = 1 - squareform(pdist(emb1, metric='cosine'))
    sim2 = 1 - squareform(pdist(emb2, metric='cosine'))

    # Flatten and compute Spearman correlation
    return spearmanr(sim1.flatten(), sim2.flatten()).correlation
