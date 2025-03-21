import networkx as nx
import matplotlib.pyplot as plt
import math
import random
import copy

### Graph for 4a, 4b and 4c: 
"""
G = nx.Graph()
G.add_nodes_from(range(1, 10))
G.add_edges_from({
    (1,2), (1,8),
    (1,7), (1,9),
    (1,3), (1,6),
    (3,6), (3,4),
    (4,6), (6,5)
})

print(G.nodes)
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
"""
##read in the Zachary Karate Club graph FOR 4D 
G = nx.read_gml("karate.gml", label=None)
G = nx.convert_node_labels_to_integers(G, first_label=1)
print(G.nodes)

def DCSBM_loglikelihood(G, part):
    groups = set(part.values())

    sum_k = {r: 0 for r in groups}
    sum_o = {(r,s): 0 for r in groups for s in groups}

    degrees = dict(G.degree())

    for node in G.nodes():
        r = part[node]
        sum_k[r] += degrees[node]
    
    # accumulate omega
    for (i, j) in G.edges():
        r = part[i]
        s = part[j]
        if r == s:
            sum_o[(r, s)] += 2
        else:
            sum_o[(r, s)] += 1
            sum_o[(s, r)] += 1
    
    logL = 0.0
    for r in groups:
        for s in groups:
            w_rs = sum_o[(r, s)]
            if w_rs > 0:
                denom = float(sum_k[r]) * float(sum_k[s])
                ratio = w_rs / denom
                logL += w_rs * math.log(ratio)
    return logL
    


def makeAMove(G, zt, c, f):
    curlogL = DCSBM_loglikelihood(G, zt)
    bestlogL = curlogL
    bestMove = (None, None)


    for node in G.nodes:
        if f[node-1] == 0:
            g1 = zt[node]

            for g2 in range(1, c+1):
                if g1 == g2:
                    continue
                zt[node] = g2
                nlogL = DCSBM_loglikelihood(G, zt)

                if nlogL > bestlogL:
                    bestlogL = nlogL
                    bestMove = (node, g2)
                zt[node] = g1 # hint 
    return bestlogL, bestMove

def runOnePhase(G, z0, c):
    z0 = {}
    for node in G.nodes():
        z0[node] = random.randint(1, c)
    zt = copy.deepcopy(z0)
    f = [0] * len(G.nodes)
    logLs = [DCSBM_loglikelihood(G, zt)]

    for move in range(len(G.nodes)):
        bestLogL, (bestNode, bestNewGroup) = makeAMove(G, zt, c, f)
        if bestNode is not None:
            zt[bestNode] = bestNewGroup
            f[bestNode - 1] = 1
            logLs.append(bestLogL)

    Lst = logLs[-1]
    zst = copy.deepcopy(zt)

    #stop = int(zst == z0)

    return zst, Lst, logLs

def fitDCSBM(G, c, T):
    z_best = {}
    for node in G.nodes():
        z_best[node] = random.randint(1, c)
    L_best = DCSBM_loglikelihood(G, z_best)

    allLogLs = []  
    phase_lls = []
    pc = 0

    for phase in range(T):
        zst, Lst, logLs = runOnePhase(G, z_best, c)

        allLogLs.extend(logLs)
        phase_lls.append(Lst)
    
        if Lst > L_best:
            z_best = copy.deepcopy(zst)
            L_best = Lst
            pc += 1
            
        #if stop ==1:
           # break

    return z_best, L_best, pc, allLogLs, phase_lls


c = 2    # 2 groups
T = 300

z_fin, L_fin, pc, log_list, phase_lls = fitDCSBM(G, c, T)

print("\n=== Results ===")
print(f"Final Partition: {z_fin}")
print(f"Log-Likelihood: {L_fin:.2f}")
print(f"Phases used: {pc}")


pos = nx.spring_layout(G, seed=42)
node_colors = []
for node in G.nodes():
    if z_fin[node] == 1:
        node_colors.append("lightblue")
    else:
        node_colors.append("green")

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")
plt.figure()
nx.draw_networkx(G, pos, labels=z_fin, node_color=node_colors)
plt.title(f"Zachary Karate Club: DC-SBM logL={L_fin:.2f}")
plt.axis("off")
plt.show()

plt.figure()
plt.plot(range(len(log_list)), log_list, marker='o', linestyle='-')
plt.xlabel("Iteration (across all phases)")
plt.ylabel("DC-SBM Log-Likelihood")
plt.title("Log-Likelihood Evolution")
plt.show()

plt.plot(range(1, len(phase_lls)+1), phase_lls, marker='o')
plt.xlabel("Phase")
plt.ylabel("DC-SBM Log-Likelihood")
plt.title("DCSBM Log-Likelihood Evolution")
plt.show()

def compute_o_k(G, partition):
        groups = set(partition.values())
        sum_k = {r:0 for r in groups}
        sum_o = {(r,s):0 for r in groups for s in groups}
        degs = dict(G.degree())

        for node in G.nodes():
            r = partition[node]
            sum_k[r] += degs[node]

        for (i,j) in G.edges():
            r = partition[i]
            s = partition[j]
            if r == s:
                sum_o[(r,s)] += 2
            else:
                sum_o[(r,s)] += 1
                sum_o[(s,r)] += 1

        return sum_o, sum_k

omega_fin, kappa_fin = compute_o_k(G, z_fin)

print("Final w matrix (each row r, col s, in sorted order of group):")
sorted_groups = sorted(list(set(z_fin.values())))
for r in sorted_groups:
    row_vals = []
    for s in sorted_groups:
        row_vals.append(omega_fin[(r,s)])
    print(f"group {r} -> {row_vals}")

print("\nFinal κ vector:")
for r in sorted_groups:
    print(f"group{r}: {kappa_fin[r]}")

"""
c = 3
T = 30
z0 = {}
for node in G.nodes():
    z0[node] = random.randint(1, c)

z_fin, L_fin, pc, log_list = fitDCSBM(G, c, T)

print(f"Final Partition z*: {z_fin}")
print(f"Final Log-Likelihood L*: {L_fin:.2f}")
print(f"Number of phases used (pc): {pc}")

pos = nx.spring_layout(G, seed=42)
node_colors = []
for node in G.nodes():
    g = z_fin[node]
    if g == 1:
        node_colors.append("gold")
    elif g == 2:
        node_colors.append("lightblue")
    else:
        node_colors.append("green")

plt.figure()
nx.draw_networkx(G, pos, labels=z_fin, with_labels=True, node_color=node_colors)
plt.title(f"Final partition z*, DC-SBM logL={L_fin:.2f}")
plt.axis("off")
plt.show()

    # Log-likelihood evolution across all phases
plt.figure()
plt.plot(range(len(log_list)), log_list, marker='o', linestyle=None)
plt.xlabel("Iteration (across all phases)")
plt.ylabel("DC-SBM Log-Likelihood")
plt.title("DC-SBM Log-Likelihood Evolution")
plt.show()

def compute_o_k(G, partition):
        groups = set(partition.values())
        sum_k = {r:0 for r in groups}
        sum_o = {(r,s):0 for r in groups for s in groups}
        degs = dict(G.degree())

        for node in G.nodes():
            r = partition[node]
            sum_k[r] += degs[node]

        for (i,j) in G.edges():
            r = partition[i]
            s = partition[j]
            if r == s:
                sum_o[(r,s)] += 2
            else:
                sum_o[(r,s)] += 1
                sum_o[(s,r)] += 1

        return sum_o, sum_k

o_fin, k_fin = compute_o_k(G, z_fin)

print("Final w matrix (each row r, col s, in sorted order of group):")
sorted_groups = sorted(list(set(z_fin.values())))
for r in sorted_groups:
    row_vals = []
    for s in sorted_groups:
        row_vals.append(o_fin[(r,s)])
    print(f"group {r} -> {row_vals}")

print("Final k vector:")
for r in sorted_groups:
    print(f"group
           {r}: {k_fin[r]}")
"""

### Code below was used for 4b plotting and running runOnePhase
"""
c = 3
z0 = {}
for node in G.nodes():
    z0[node] = random.randint(1, c)

z_star, L_star, h, logL_values = runOnePhase(G, z0, c)

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

plt.figure()
pos = nx.spring_layout(G, seed=42)
node_colors = ["gold" if z0[node] == 1 else "lightblue" if z0[node] == 2 else "green" for node in G.nodes()]
nx.draw_networkx(G, pos, labels=z0, with_labels=True, node_color=node_colors)
plt.title(f"Initial partition z0, DC-SBM logL={logL_values[0]:.2f}")
plt.axis("on")
plt.show()

plt.figure()
node_colors = ["gold" if z_star[node] == 1 else "lightblue" if z_star[node] == 2 else "green" for node in G.nodes()]
nx.draw_networkx(G, pos, labels=z_star, with_labels=True, node_color=node_colors)
plt.title(f"Final partition z*, DC-SBM logL={L_star:.2f}")
plt.axis("on")
plt.show()

plt.figure()
plt.plot(range(len(logL_values)), logL_values, marker='o', linestyle='-', color='green')
plt.xlabel("Iteration")
plt.ylabel("DC-SBM Log-Likelihood")
plt.title("Evolution of DC-SBM Log-Likelihood (runOnePhase)")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(False)
plt.show()

print(f"Final Log-Likelihood: {L_star:.2f}")
print(f"Halting Indicator (1=halt, 0=continue): {h}")
print(f"Best Partition: {z_star}")
"""

###Code below was used for Question 4a plotting and Make a Move inputs
"""
c = 2
zt = {}
for node in G.nodes():
    zt[node] = random.randint(1, c)

f = [0,0,0,0,0,0,0,0,0]
f[1] = 1

starting_LogL = DCSBM_loglikelihood(G, zt)
plt.figure()
pos = nx.spring_layout(G, seed=42)
node_colors = []
for node in G.nodes():
    if zt[node] == 1:
        node_colors.append("lightblue")  # Group 1 nodes are blue
    else:
        node_colors.append("green")  # Group 2 nodes are green
nx.draw_networkx(G, pos, labels=zt, with_labels=True, node_color=node_colors)
plt.title(f"Initial partition zt, DC-SBM logL={starting_LogL:.2f}")
plt.axis("on")
plt.show()


bestLogL, (bestNode, bestNewGroup) = makeAMove(G, zt, c, f)

if bestNode is not None:
    zt[bestNode] = bestNewGroup
    # freeze that node
    f[bestNode - 1] = 1

new_LogL = DCSBM_loglikelihood(G, zt)
plt.figure()
node_colors = []
for node in G.nodes():
    if zt[node] == 1:
        node_colors.append("lightblue")  # Group 1 nodes are blue
    else:
        node_colors.append("green")  # Group 2 nodes are green
nx.draw_networkx(G, pos, labels=zt, with_labels=True, node_color=node_colors)
plt.title(f"After best move: node {bestNode}->{bestNewGroup}, DC-SBM logL={new_LogL:.2f}")
plt.axis("on")
plt.show()
"""