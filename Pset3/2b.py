import numpy as np
import pandas as pd
import networkx as nx
import random
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score

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
    return G

# make G' grom G by removing fraction f of edges (validation graph)
def generate_validation_graph(G, f):

    all_edges = list(G.edges())
    num_edges = len(all_edges)
    num_obs = int(round(f * num_edges))
    obs_edges = random.sample(all_edges, num_obs)
    G_val = nx.Graph()
    G_val.add_nodes_from(G.nodes())
    G_val.add_edges_from(obs_edges)
    # missing edges are edges in G but not in G'
    removed_edges = set(all_edges) - set(obs_edges)
    return G_val, list(removed_edges)

# make G'' from G' by sampling fraction f of edges (training graph)
def generate_training_graph(G_val, f):
    val_edges = list(G_val.edges())
    num_val_edges = len(val_edges)
    num_obs = int(round(f * num_val_edges))
    obs_edges = random.sample(val_edges, num_obs)
    G_train = nx.Graph()
    G_train.add_nodes_from(G_val.nodes())
    G_train.add_edges_from(obs_edges)
    removed_edges = set(val_edges) - set(obs_edges)  # edges in G' but not in G''
    return G_train, list(removed_edges)

# functions for single predictors as used in 2a 
def dp_score(i, j, G_obs):
    return G_obs.degree(i) * G_obs.degree(j)

def jc_score(i, j, G_obs):
    N_i = set(G_obs.neighbors(i))
    N_j = set(G_obs.neighbors(j))
    union = N_i.union(N_j)
    if not union:
        return 0
    return len(N_i.intersection(N_j)) / len(union)

def sp_score(i, j, G_obs, epsilon=1e-6):
    try:
        length = nx.shortest_path_length(G_obs, source=i, target=j)
        return 1/length + np.random.uniform(0, epsilon)
    except nx.NetworkXNoPath:
        return 0

# build features of predictors scores for a given set of edges to use in validation matrix or training matrix 
def build_features(G_obs, edges, label, epsilon=1e-6):
    data = []
    for (i, j) in edges:
        dp_val = dp_score(i, j, G_obs)
        jc_val = jc_score(i, j, G_obs)
        sp_val = sp_score(i, j, G_obs, epsilon) 
        data.append([dp_val, jc_val, sp_val, label])
    return data

# make validation matrix S from G' and G by taking all pairs not in G' and labeling them 1 if in G and 0 if not
def build_validation_data(G_val, G, missing_edges, epsilon=1e-6):
    X_val = list(nx.non_edges(G_val))
    pos_val = [(i,j) for (i,j) in X_val if (i,j) in missing_edges or (j,i) in missing_edges]
    neg_val = [(i,j) for (i,j) in X_val if (i,j) not in missing_edges and (j,i) not in missing_edges]
    data_pos = build_features(G_val, pos_val, 1, epsilon) # use build features function to get the scores for the pairs
    data_neg = build_features(G_val, neg_val, 0, epsilon)
    return data_pos + data_neg

# make training matrix T using c pairs from true positive and true negative edges in G''
def build_training_data(G_train, pos_edges, c=1000, epsilon=1e-6):
    if len(pos_edges) < c:
        chosen_pos = [random.choice(pos_edges) for _ in range(c)]
    else:
        chosen_pos = random.sample(pos_edges, c)
    all_non_edges = list(nx.non_edges(G_train))
    if len(all_non_edges) < c:
        chosen_neg = [random.choice(all_non_edges) for _ in range(c)]
    else:
        chosen_neg = random.sample(all_non_edges, c)
    data_pos = build_features(G_train, chosen_pos, 1, epsilon) # use build features function to get the scores for the pairs
    data_neg = build_features(G_train, chosen_neg, 0, epsilon)
    return data_pos + data_neg

# stacked edge prediction experiment
def stacked_edge_prediction_experiment(G, f, c=1000, epsilon=1e-6, random_state=0):
    from sklearn.ensemble import RandomForestClassifier
    random.seed(random_state)
    np.random.seed(random_state)
    # make G'
    G_val, missing_edges = generate_validation_graph(G, f)
    # make G''
    G_train, train_pos = generate_training_graph(G_val, f)
    # make matrix T
    train_data = build_training_data(G_train, train_pos, c=c, epsilon=epsilon)
    X_train = [row[:3] for row in train_data]
    y_train = [row[3] for row in train_data]

    # Train random forest
    rf = RandomForestClassifier(n_estimators=300, random_state=random_state)
    rf.fit(X_train, y_train)

    #make validation matrix S
    val_data = build_validation_data(G_val, G, missing_edges, epsilon=epsilon)
    X_val = [row[:3] for row in val_data]
    y_val = [row[3] for row in val_data]

    if len(set(y_val)) < 2:
        return 0.5  # return 0.5 if only one class is present

    y_scores = rf.predict_proba(X_val)[:,1]
    stacked_auc = roc_auc_score(y_val, y_scores)
    return stacked_auc

#experiment function for supervised approach through f values
def run_supervised_experiment(G, f_values, c=1000, n_reps=10):

    results = {'f': [], 'stacked_mean': [], 'stacked_std': [], 'dp_mean': [], 'dp_std': [], 'jc_mean': [], 'jc_std': [], 'sp_mean': [], 'sp_std': []}

    for f in f_values:
        print(f"Running supervised experiment at f={f:.2f}")
        stack_aucs = []
        dp_aucs = []
        jc_aucs = []
        sp_aucs = []

        for rep in range(n_reps):
            # measure auc for stacked classifier
            auc_stacked = stacked_edge_prediction_experiment(G, f, c=c, random_state=rep)
            stack_aucs.append(auc_stacked)
            # measure auc for individual predictors
            G_val, missing_edges = generate_validation_graph(G, f)
            non_edges_val = list(nx.non_edges(G_val))
            # get positive and negative edges
            pos_edges = [(i,j) for (i,j) in non_edges_val if (i,j) in missing_edges or (j,i) in missing_edges]
            neg_edges = [(i,j) for (i,j) in non_edges_val if (i,j) not in missing_edges and (j,i) not in missing_edges]

            # Avoid class imbalance by sampling negative edges bc there are more negative edges than positive
            neg_edges_samp = random.sample(neg_edges, min(len(neg_edges), len(pos_edges)))

            cand = []
            # build features for each pair
            def feat_row(i,j,label):
                return [dp_score(i,j,G_val), jc_score(i,j,G_val), sp_score(i,j,G_val), label]
            for (i,j) in pos_edges: # add positive edges
                cand.append(feat_row(i,j,1))
            for (i,j) in neg_edges_samp: # add negative edges
                cand.append(feat_row(i,j,0))
            if len(cand)==0 or len(set(r[3] for r in cand))<2: # skip if only positive or only negative 
                dp_aucs.append(0.5)
                jc_aucs.append(0.5)
                sp_aucs.append(0.5)
            else:
                #compute auc for each predictor using roc_auc_score from sklearn
                y_true = [r[3] for r in cand]
                dp_s = [r[0] for r in cand]
                jc_s = [r[1] for r in cand]
                sp_s = [r[2] for r in cand]
                dp_aucs.append(roc_auc_score(y_true, dp_s))
                jc_aucs.append(roc_auc_score(y_true, jc_s))
                sp_aucs.append(roc_auc_score(y_true, sp_s))

        #take the mean and std of the aucs for each predictor and stacked classifier
        results['f'].append(f)
        results['stacked_mean'].append(np.mean(stack_aucs))
        results['stacked_std'].append(np.std(stack_aucs))
        results['dp_mean'].append(np.mean(dp_aucs))
        results['dp_std'].append(np.std(dp_aucs))
        results['jc_mean'].append(np.mean(jc_aucs))
        results['jc_std'].append(np.std(jc_aucs))
        results['sp_mean'].append(np.mean(sp_aucs))
        results['sp_std'].append(np.std(sp_aucs))

    return results

#plotting function for the results of the supervised experiment
import seaborn as sns
sns.set(style='white')
sns.set_context('talk')
def plot_stacked_results(results):
    import matplotlib.pyplot as plt
    f_vals = results['f']
    plt.figure(figsize=(8,6))
    plt.errorbar(f_vals, results['dp_mean'], yerr=results['dp_std'], fmt='-o', label='DP')
    plt.errorbar(f_vals, results['jc_mean'], yerr=results['jc_std'], fmt='-s', label='JC')
    plt.errorbar(f_vals, results['sp_mean'], yerr=results['sp_std'], fmt='-^', label='SP')
    plt.errorbar(f_vals, results['stacked_mean'], yerr=results['stacked_std'], fmt='-D', label='Stacked')
    plt.axhline(0.5, color='k', linestyle='--', label='Chance')
    plt.xlabel('Fraction f of observed edges')
    plt.ylabel('AUC')
    plt.title('Edge Prediction AUC vs. f (Malaria Network)')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()
    plt.show()

#function to get the roc curve for the stacked classifier and the individual predictors
def plot_stacked_roc(G, f=0.8, c=1000):
    from sklearn.ensemble import RandomForestClassifier
    G_val, missing_edges = generate_validation_graph(G, f)
    non_edges_val = list(nx.non_edges(G_val))
    pos_edges = [(i,j) for (i,j) in non_edges_val if (i,j) in missing_edges or (j,i) in missing_edges]
    neg_edges = [(i,j) for (i,j) in non_edges_val if (i,j) not in missing_edges and (j,i) not in missing_edges]
    neg_edges_samp = random.sample(neg_edges, min(len(neg_edges), len(pos_edges)))

    cand = []
    # build features for each pair as done prior 
    def feat_row(i,j,label):
        return [dp_score(i,j,G_val), jc_score(i,j,G_val), sp_score(i,j,G_val), label]
    for (i,j) in pos_edges:
        cand.append(feat_row(i,j,1))
    for (i,j) in neg_edges_samp:
        cand.append(feat_row(i,j,0))

    X_val = [row[:3] for row in cand]
    y_val = [row[3] for row in cand]

    #Get the training graph and training data from G'
    G_train, train_pos = generate_training_graph(G_val, f)
    train_data = build_training_data(G_train, train_pos, c=c)
    X_train = [row[:3] for row in train_data]
    y_train = [row[3] for row in train_data]

    #train the random forest classifier
    rf = RandomForestClassifier(n_estimators=1000, random_state=0)
    rf.fit(X_train, y_train)

    stacked_scores = rf.predict_proba(X_val)[:,1]
    dp_scores = [r[0] for r in cand]
    jc_scores = [r[1] for r in cand]
    sp_scores = [r[2] for r in cand]

    from sklearn.metrics import roc_curve
    #get the tpr and fpr for each predictor and the stacked classifier
    dp_fpr, dp_tpr, _ = roc_curve(y_val, dp_scores)
    jc_fpr, jc_tpr, _ = roc_curve(y_val, jc_scores)
    sp_fpr, sp_tpr, _ = roc_curve(y_val, sp_scores)
    stacked_fpr, stacked_tpr, _ = roc_curve(y_val, stacked_scores)

    #get the auc for each predictor and the stacked classifier
    dp_auc = auc(dp_fpr, dp_tpr)
    jc_auc = auc(jc_fpr, jc_tpr)
    sp_auc = auc(sp_fpr, sp_tpr)
    stacked_auc = auc(stacked_fpr, stacked_tpr)

    plt.figure(figsize=(7,6))
    plt.plot(dp_fpr, dp_tpr, label=f'DP (AUC={dp_auc:.2f})')
    plt.plot(jc_fpr, jc_tpr, label=f'JC (AUC={jc_auc:.2f})')
    plt.plot(sp_fpr, sp_tpr, label=f'SP (AUC={sp_auc:.2f})')
    plt.plot(stacked_fpr, stacked_tpr, label=f'Stacked (AUC={stacked_auc:.2f})')
    plt.plot([0,1],[0,1],'k--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for f={f} (Malaria Network)')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()
    plt.show()

def main():
    random.seed(0)
    np.random.seed(0)

    # Load Malariaa HVR_5  network
    malaria_edges_file   = 'data/HVR_5.txt'
    malaria_metadata_file = 'data/metadata_CysPoLV.txt'
    G_mal = load_malaria_hvr_data(malaria_edges_file, malaria_metadata_file)

    f_values = [0.5, 0.6, 0.7, 0.8, 0.9] # descrete f to test
    results = run_supervised_experiment(G_mal, f_values, c=1000, n_reps=10)
    plot_stacked_results(results)

    # ROC at f=0.8
    plot_stacked_roc(G_mal, f=0.8, c=1000)

if __name__=='__main__':
    main()