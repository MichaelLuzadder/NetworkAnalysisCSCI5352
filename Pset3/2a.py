import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_auc_score, roc_curve, auc

# Read the Norwegian Board Network
def load_norwegian_board_data(nodes_file, edges_file):
    #read nodes from metadata
    df_nodes = pd.read_csv(nodes_file, skiprows=1, header=None, names=['index','vid','name','gender','_pos'])

    node_ids = df_nodes['index'].values
    labels = df_nodes['gender'].values  

    # Create a dictionary of node and labels
    true_labels = {int(nid): lab for nid, lab in zip(node_ids, labels)}

    # Read edges
    df_edges = pd.read_csv(edges_file, skiprows=1, header=None, names=['source','target'])
    edges = df_edges[['source','target']].values

    # make networkx graph
    G = nx.Graph()
    G.add_nodes_from(node_ids)  
    for s, t in edges:
        G.add_edge(s, t)

    return G, true_labels

# Read the Malaria Network
def load_malaria_hvr_data(edges_file, metadata_file):
    #Read metadata this file is just of the attributes: im assuming the order is the same as the nodes??
    with open(metadata_file, 'r') as f:
        lines = f.read().strip().split('\n')
    labels = [int(x) for x in lines]

    # make dictionary of nodes and lebels
    true_labels = {}
    for i, lab in enumerate(labels): # make sure this matches the node order
        true_labels[i] = lab  

    # Read edges
    edges = []
    with open(edges_file, 'r') as f:
        for line in f:
            if line.strip():
                s_str, t_str = line.strip().split(',')
                s = int(s_str) - 1
                t = int(t_str) - 1
                edges.append((s, t))

    # make networkx graph
    all_nodes = list(true_labels.keys()) # make sure this matches the node order
    G = nx.Graph()
    G.add_nodes_from(all_nodes)
    G.add_edges_from(edges)

    return G, true_labels

# make G observed from G
def generate_observed_graph(G, f):

    all_edges = list(G.edges())
    num_edges = len(all_edges)
    num_obs = int(round(f * num_edges))
    obs_edges = random.sample(all_edges, num_obs)    # Randomly select observed edges
    G_obs = nx.Graph()
    G_obs.add_nodes_from(G.nodes())
    G_obs.add_edges_from(obs_edges)

    removed_edges = set(all_edges) - set(obs_edges) # get the removed edges
    return G_obs, list(removed_edges)

# edge predictors 
def dp_score(i, j, G_obs):
    return G_obs.degree(i) * G_obs.degree(j)

def jc_score(i, j, G_obs):
    neighbors_i = set(G_obs.neighbors(i))
    neighbors_j = set(G_obs.neighbors(j))
    union = neighbors_i.union(neighbors_j)
    if len(union) == 0: # Avoid division by zero
        return 0 
    return len(neighbors_i.intersection(neighbors_j)) / len(union)

def sp_score(i, j, G_obs, epsilon=1e-6):
    try:
        sp_length = nx.shortest_path_length(G_obs, source=i, target=j)
        return 1 / sp_length + np.random.uniform(0, epsilon) # Add a tiny random offset
    except nx.NetworkXNoPath: # i and j are disconnected
        return 0

# create candidate table
def create_candidate_table(G, G_obs, positive_edges, n_negatives=None, epsilon=1e-6, tie_epsilon=1e-9):
    #tau here is the label column
    table = []
    # positive examples
    for (i, j) in positive_edges:
        dp_val = dp_score(i, j, G_obs)
        jc_val = jc_score(i, j, G_obs)
        sp_val = sp_score(i, j, G_obs, epsilon)
        #  break ties
        dp_val += random.uniform(0, tie_epsilon)
        jc_val += random.uniform(0, tie_epsilon)
        sp_val += random.uniform(0, tie_epsilon)
        row = {'i': i, 'j': j, 'dp': dp_val, 'jc': jc_val, 'sp': sp_val, 'label': 1}
        table.append(row)
    n_pos = len(positive_edges)
    
    # Negative examples
    if n_negatives is None:
        n_negatives = n_pos
    all_non_edges = list(nx.non_edges(G))
    negative_edges = random.sample(all_non_edges, n_negatives)

    for (i, j) in negative_edges:
        dp_val = dp_score(i, j, G_obs)
        jc_val = jc_score(i, j, G_obs)
        sp_val = sp_score(i, j, G_obs, epsilon)
        # =break ties
        dp_val += random.uniform(0, tie_epsilon)
        jc_val += random.uniform(0, tie_epsilon)
        sp_val += random.uniform(0, tie_epsilon)
        row = {'i': i, 'j': j, 'dp': dp_val, 'jc': jc_val, 'sp': sp_val, 'label': 0}
        table.append(row)
    return table

# compute AUC
def compute_auc(candidate_table, score_key):
    # use scikit-learn's roc_auc_score which expects y_true and y_scores
    y_true = [row['label'] for row in candidate_table]
    y_scores = [row[score_key] for row in candidate_table]
    if len(set(y_true)) < 2:
        return 0.5 # default to chance if len(set(y_true)) < 2
    return roc_auc_score(y_true, y_scores)

# run link prediction for one f
def run_link_prediction_experiment(G, f, n_reps=100, epsilon=1e-6):

    auc_dp_list = []
    auc_jc_list = []
    auc_sp_list = []
    
    for _ in range(n_reps):
        G_obs, removed_edges = generate_observed_graph(G, f)
        candidate_table = create_candidate_table(G, G_obs, removed_edges, epsilon=epsilon)
        auc_dp = compute_auc(candidate_table, 'dp')
        auc_jc = compute_auc(candidate_table, 'jc')
        auc_sp = compute_auc(candidate_table, 'sp')
        auc_dp_list.append(auc_dp)
        auc_jc_list.append(auc_jc)
        auc_sp_list.append(auc_sp)
    return (np.mean(auc_dp_list), np.std(auc_dp_list), np.mean(auc_jc_list), np.std(auc_jc_list), np.mean(auc_sp_list), np.std(auc_sp_list))

# run experiment over f
def run_experiment_over_f(G, f_values, n_reps=100, epsilon=1e-6):
    results = {'f': [], 'dp_mean': [], 'dp_std': [], 'jc_mean': [], 'jc_std': [], 'sp_mean': [], 'sp_std': []}
    for f in f_values:
        print(f"Running experiment for f = {f:.2f}")
        dp_mean, dp_std, jc_mean, jc_std, sp_mean, sp_std = run_link_prediction_experiment(G, f, n_reps, epsilon)
        results['f'].append(f)
        results['dp_mean'].append(dp_mean)
        results['dp_std'].append(dp_std)
        results['jc_mean'].append(jc_mean)
        results['jc_std'].append(jc_std)
        results['sp_mean'].append(sp_mean)
        results['sp_std'].append(sp_std)
    return results

# plot roc curves
import seaborn as sns
sns.set(style='white')
sns.set_context('talk')
def plot_roc_curves(G, f, epsilon=1e-6):
    G_obs, removed_edges = generate_observed_graph(G, f)
    candidate_table = create_candidate_table(G, G_obs, removed_edges, epsilon=epsilon)
    y_true = [row['label'] for row in candidate_table]
    # get scores and compute fpr, tpr.
    predictors = ['dp', 'jc', 'sp']
    colors = {'dp': 'skyblue', 'jc': 'olivedrab', 'sp': 'lightcoral'} 
    plt.figure(figsize=(8,6))
    for pred in predictors:
        y_scores = [row[pred] for row in candidate_table]
        if len(set(y_true)) < 2: # skip if only one class
            continue
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{pred.upper()} (AUC = {roc_auc:.2f})", color=colors[pred])
    plt.plot([0,1],[0,1],'k--', label='Chance (AUC = 0.5)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for f = {f:.2f}')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()
    plt.show()

def main():
    random.seed(0)
    np.random.seed(0)
    #load the data
    norwegian_nodes_file = 'data/net1m_2011-08-01.csv/nodes.csv'
    norwegian_edges_file = 'data/net1m_2011-08-01.csv/edges.csv'
    G_nb, true_labels_nb = load_norwegian_board_data(norwegian_nodes_file, norwegian_edges_file)  
    malaria_edges_file   = 'data/HVR_5.txt'
    malaria_metadata_file = 'data/metadata_CysPoLV.txt'
    G_mal, true_labels_mal = load_malaria_hvr_data(malaria_edges_file, malaria_metadata_file)
    #fraction of observed edges f
    f_values = np.arange(0.05, 1.00, 0.05)
    n_reps = 100  # number of repetitions per f

    print("Running link prediction experiments on Norwegian Board network...")
    results_nb = run_experiment_over_f(G_nb, f_values, n_reps)
    print("Running link prediction experiments on Malaria HVR_5 network...")
    results_mal = run_experiment_over_f(G_mal, f_values, n_reps)
    
    #plot AUC vs f for main experiment
    # Norwegian Board 
    plt.figure(figsize=(8,6))
    plt.errorbar(results_nb['f'], results_nb['dp_mean'], yerr=results_nb['dp_std'], fmt='-o', capsize=4, label='DP', color='skyblue')
    plt.errorbar(results_nb['f'], results_nb['jc_mean'], yerr=results_nb['jc_std'], fmt='-s', capsize=4, label='JC', color='olivedrab')
    plt.errorbar(results_nb['f'], results_nb['sp_mean'], yerr=results_nb['sp_std'], fmt='-^', capsize=4, label='SP', color='lightcoral')
    plt.axhline(0.5, color='k', linestyle='--', label='Chance (AUC=0.5)')
    plt.xlabel('Fraction f of observed edges')
    plt.ylabel('AUC')
    plt.title('Link Prediction AUC vs. f(Norwegian Board)')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()
    plt.show()
    
    # Malaria HVR_5
    plt.figure(figsize=(8,6))
    plt.errorbar(results_mal['f'], results_mal['dp_mean'], yerr=results_mal['dp_std'], fmt='-o', capsize=4, label='DP', color='skyblue')
    plt.errorbar(results_mal['f'], results_mal['jc_mean'], yerr=results_mal['jc_std'], fmt='-s', capsize=4, label='JC', color='olivedrab')
    plt.errorbar(results_mal['f'], results_mal['sp_mean'], yerr=results_mal['sp_std'], fmt='-^', capsize=4, label='SP', color='lightcoral')
    plt.axhline(0.5, color='k', linestyle='--', label='Chance (AUC=0.5)')
    plt.xlabel('Fraction f of observed edges')
    plt.ylabel('AUC')
    plt.title('Link Prediction AUC vs. f(Malaria HVR_5)')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()
    plt.show()

# for ROC curves 
    f_target = 0.8
    print(f"Plotting ROC curves for Norwegian Board network at f = {f_target}")
    plot_roc_curves(G_nb, f_target)
    print(f"Plotting ROC curves for Malaria network at f = {f_target}")
    plot_roc_curves(G_mal, f_target)
if __name__ == '__main__':
    main()