import networkx as nx
import matplotlib.pyplot as plt
import math
import random
import copy

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

c = 3
T = 30


z_fin, L_fin, pc, log_list, phase_lls = fitDCSBM(G, c, T)

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

import seaborn as sns 
sns.set_style("white")
sns.set_context("talk")
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
    print(f"group{r}: {k_fin[r]}")