import pandas as pd

# threshold for the SCN file to make it binary
## This was necessary bc the file was needed to be thresholded using values in the github. 
## this is related to question 4 for the connectome network of mouse SCN
threshold = 0.949

# Load the MIC adjacency matrix withouth headers
mic_scores = pd.read_csv("NetworkAnalysisCSCI5352/Pset2/mic_indiv_scn1_Abel2016_BiologicalConnectome.csv", header=None)

# Apply threshold to create a binary adjacency matrix
network = (mic_scores >= threshold).astype(int)

network.to_csv("scn1_thresholded.csv", header=False, index=False)
