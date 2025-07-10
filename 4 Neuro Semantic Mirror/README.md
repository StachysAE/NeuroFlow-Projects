# Neuro Semantic Mirror

An exploratory project aimed at understanding how the brain represents semantic concepts by analyzing fMRI data. This module attempts to map neural activity patterns to language-based meaning, serving as a step toward cognitive decoding and neuro-symbolic representation.

---

## Purpose

- Explore how semantic information is encoded in the brain  
- Work with open-source fMRI datasets for natural language processing tasks  
- Experiment with mapping brain activity to word embeddings or conceptual categories  

---

## Features

- Prepares and preprocesses fMRI datasets using BIDS structure  
- Extracts brain region activations related to semantic stimuli  
- Attempts basic decoding using PCA, regression, and classification techniques  
- Integrates external semantic embeddings (e.g., GloVe or word2vec)

---

## Technologies Used

- Python  
- NiBabel, Nilearn (for fMRI data handling)  
- Scikit-learn, NumPy, Pandas  
- Gensim or spaCy (for semantic vectors)

---

## Note

- fMRI datasets are not included in the repo due to size; please download them from OpenNeuro.
- This is an early-stage experimental module and may not yield robust decoding results — it's meant for learning and exploration.

---
