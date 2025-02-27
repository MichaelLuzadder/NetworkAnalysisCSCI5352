import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random

# for helping with baseline prediction 
def compute_class_distribution(true_labels):

    from collections import Counter
    label_counts = Counter(true_labels.values())
    total_nodes = len(true_labels)
    class_probs = {lab: count / total_nodes for lab, count in label_counts.items()}
    return class_probs

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


# Local Smoothing generalized to work with any graph
def local_smoothing_predict(G, true_labels, labeled_nodes):
    # entire network label distribution for baseline 
    class_probs = compute_class_distribution(true_labels)
    global_labels = list(class_probs.keys())
    global_weights = list(class_probs.values())

    predicted_labels = {}
    for node in G.nodes():
        if node in labeled_nodes:
            # if label already exits
            predicted_labels[node] = true_labels[node]
        else:
            # if its unlabeled look at neighbors
            neighbor_labels = [true_labels[nbr] for nbr in G[node] if nbr in labeled_nodes]
            if not neighbor_labels:
                # make baseline prediction if no neighbors are labeled
                fallback_label = random.choices(global_labels, weights=global_weights, k=1)[0]
                predicted_labels[node] = fallback_label
            else:
                # find the mode of the neighbors
                counts = {}
                for lab in neighbor_labels:
                    counts[lab] = counts.get(lab, 0) + 1
                max_count = max(counts.values())
                # use max count for neighbors then randomly break ties
                best_labs = [lab for lab, c in counts.items() if c == max_count]
                predicted_labels[node] = random.choice(best_labs) # could do this with addition of small number to break ties but i think this also works 

    return predicted_labels

# Measure Accuracy
def measure_accuracy(true_labels, predicted_labels, unlabeled_nodes):

    total = len(unlabeled_nodes)
    if total == 0:
        return 1.0  # handle when alpha = 1.0

    correct = sum(true_labels[node] == predicted_labels[node] for node in unlabeled_nodes)
    return correct / total

# Experiment function given a graph and true labels alpha values and number of repitions

def run_experiment(G, true_labels, alpha_values, n_reps=1000):
    all_nodes = list(G.nodes())
    num_nodes = len(all_nodes)
    accs_mean = []
    accs_std = []

    for alpha in alpha_values:
        k = int(np.round(alpha * num_nodes))  # round to nearest integer for number of labeled nodes
        rep_accuracies = []

        for _ in range(n_reps):
            labeled_nodes = set(random.sample(all_nodes, k)) # randomly sample k nodes to be labeled (alpha fraction)
            pred_labels = local_smoothing_predict(G, true_labels, labeled_nodes) # predict labels for unlabeled nodes
            unlabeled_nodes = [n for n in all_nodes if n not in labeled_nodes] # get unlabeled nodes
            acc = measure_accuracy(true_labels, pred_labels, unlabeled_nodes) # measure accuracy
            rep_accuracies.append(acc) 

        accs_mean.append(np.mean(rep_accuracies)) # average accuracy over n_reps
        accs_std.append(np.std(rep_accuracies)) # standard deviation of accuracy over n_reps

    return accs_mean, accs_std


def main():
    random.seed(0)  
    np.random.seed(0)
    # load the data
    norwegian_nodes_file = 'data/net1m_2011-08-01.csv/nodes.csv'
    norwegian_edges_file = 'data/net1m_2011-08-01.csv/edges.csv'
    G_nb, true_labels_nb = load_norwegian_board_data(norwegian_nodes_file, norwegian_edges_file)
    malaria_edges_file   = 'data/HVR_5.txt'
    malaria_metadata_file = 'data/metadata_CysPoLV.txt'
    G_mal, true_labels_mal = load_malaria_hvr_data(malaria_edges_file, malaria_metadata_file)

    alpha_values = np.arange(0, 1.01, 0.01) # set range of alpha values
    n_reps = 1000 # number of repitions
    print("Running experiment on Norwegian Board network...")
    nb_acc_mean, nb_acc_std = run_experiment(G_nb, true_labels_nb, alpha_values, n_reps)
    print("Running experiment on Malaria HVR network...")
    mal_acc_mean, mal_acc_std = run_experiment(G_mal, true_labels_mal, alpha_values, n_reps)
    # plot the results
    import seaborn as sns
    sns.set(style='white')
    sns.set_context('talk')

    plt.figure(figsize=(8, 6))
    plt.errorbar(alpha_values, nb_acc_mean, yerr=nb_acc_std, label="Norwegian Board", fmt='-o', capsize=4, color='olivedrab')
    plt.errorbar(alpha_values, mal_acc_mean, yerr=mal_acc_std, label="Malaria HVR_5", fmt='-s', capsize=4, color='skyblue')
    plt.xlabel(r'Fraction $\alpha$ of labeled nodes')
    plt.ylabel('Average Accuracy ± Std Dev')
    plt.title(r'Local Smoothing Accuracy vs. Fraction Labeled ($\alpha$)')
    plt.legend()
    plt.grid(False)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

if __name__ == '__main__':
    main()