import pandas as pd
import numpy as np
import random
import networkx as nx
from copy import deepcopy
from Bio.SeqIO import FastaIO

CLUSTER_COLUMN_1 = 'cluster_1'
CLUSTER_COLUMN_2 = 'cluster_2'


# Read a .fasta file of sequences into a dictionary object
def load_sequences(filename):
    sequence_dict = {}
    with open(filename, 'r') as fastafile:
        reader = FastaIO.FastaIterator(fastafile)
        for record in reader:
            sequence_dict[record.id.split('.')[0]] = record
    return sequence_dict


# Load in a CDHit cluster file for later use in GRAND
def read_cluster_file(cluster_filename):
    #
    #   Formatted like:
    #     >Cluster 0
    #     0	15054nt, >Q81SN0... *
    #     >Cluster 1
    #     0	14874nt, >O14686... *
    #     >Cluster 2
    #     0	13773nt, >Q14517... *
    #     >Cluster 3
    #     0	13677nt, >Q9KS12... *
    #     >Cluster 4
    #     0	13644nt, >Q15149... *
    #     >Cluster 5
    #     0	13303nt, >Q96PK2... *
    #     >Cluster 6
    #     0	13137nt, >Q4JG03... *
    #     1	168nt, >Q7Z6Z7... at +/95.83%
    #
    cluster_dict = {}
    current_cluster = ''
    with open(cluster_filename, 'r') as cluster_file:
        for line in cluster_file.readlines():
            if line.startswith('>'):
                current_cluster = line.replace('>', '').replace('\n', '')
            else:
                line = line.replace('\n', '')
                start_id = line.find('>')
                end_id = line.find('.')
                cluster_dict[line[start_id + 1: end_id]] = current_cluster
    return cluster_dict


# taking in a dataframe with a row for each pair (and a column for each member of the pairs),
# and the .clstr file produced by clustering the sequences in CDHit.
# The same IDs should be used in the .fasta files passed in to CDHit and in the pairs_df dataframe.
def run_grand(cluster_dict, pairs_df, id_col1, id_col2):
    # add cluster information to pair dataframe
    clusters_1 = [cluster_dict[x] if x in cluster_dict else '' for x in pairs_df[id_col1]]
    clusters_2 = [cluster_dict[x] if x in cluster_dict else '' for x in pairs_df[id_col2]]
    pairs_df[CLUSTER_COLUMN_1] = clusters_1
    pairs_df[CLUSTER_COLUMN_2] = clusters_2
    # Drop rows that aren't in the cluster
    pairs_df = pairs_df.query(f'{CLUSTER_COLUMN_1} != "" and {CLUSTER_COLUMN_2} != ""')
    # Drop self-clusters
    pairs_df = pairs_df.query(f'{CLUSTER_COLUMN_1} != {CLUSTER_COLUMN_2}')
    unique_ids = list(
        np.unique(list(np.unique(pairs_df[CLUSTER_COLUMN_1])) + list(np.unique(pairs_df[CLUSTER_COLUMN_2]))))
    g = nx.Graph()
    g.add_nodes_from(unique_ids)
    for i in range(len(pairs_df)):
        ida = pairs_df.iloc[i][CLUSTER_COLUMN_1]
        idb = pairs_df.iloc[i][CLUSTER_COLUMN_2]
        g.add_edge(ida, idb)
    loop_count = 0
    random.seed(1)
    np.random.seed(1)
    g_copy = deepcopy(g)
    nodes, degrees = zip(*g_copy.degree)
    num_edges = 0
    new_num_edges = 1
    # Continue until no nodes with multiple edges remain
    while np.max(degrees) > 1 and loop_count < 1000:
        # First priority: find nodes with degree 1. In random order, find the nodes to which they connect
        # and drop all of those nodes' edges. Then drop any nodes that are left without any edges
        while new_num_edges != num_edges:
            num_edges = new_num_edges
            inds_1 = np.where(np.array(degrees) == 1)[0]
            random.shuffle(inds_1)
            nodes_1 = np.array(nodes)[inds_1]
            for node in nodes_1:
                for neighbour in g_copy.neighbors(node):
                    neighbours = [n for n in g_copy.neighbors(neighbour)]
                    if len(neighbours) > 1:
                        for neighbour_2 in neighbours:
                            if neighbour_2 != node:
                                g_copy.remove_edge(neighbour, neighbour_2)
            new_num_edges = len(g_copy.edges)
            nodes, degrees = zip(*g_copy.degree)
        # When no nodes with degree 1 remain, find the edge with the lowest sum of node degrees and drop
        # those nodes' other neighbouring edges
        edges = [e for e in g_copy.edges()]
        edge_sums = np.zeros((len(edges),))
        for i in range(len(edges)):
            edge = edges[i]
            sum_degree = len([n for n in g_copy.neighbors(edge[0])]) + len([n for n in g_copy.neighbors(edge[1])])
            edge_sums[i] = sum_degree
        min_edge_sum = -1
        for j in range(3, int(np.max(edge_sums)) + 1):
            if len(np.where(np.array(edge_sums) == j)[0]) != 0:
                min_edge_sum = j
                break
        inds_min = np.where(np.array(edge_sums) == min_edge_sum)[0]
        if len(inds_min) > 0:
            random.shuffle(inds_min)
            edge = edges[inds_min[0]]
            neighbours = [n for n in g_copy.neighbors(edge[0])]
            for neighbour_2 in neighbours:
                if neighbour_2 != edge[1]:
                    g_copy.remove_edge(edge[0], neighbour_2)
            neighbours = [n for n in g_copy.neighbors(edge[1])]
            for neighbour_2 in neighbours:
                if neighbour_2 != edge[0]:
                    g_copy.remove_edge(edge[1], neighbour_2)
        # drop orphaned nodes:
        nodes, degrees = zip(*g_copy.degree)
        for node, degree in zip(nodes, degrees):
            if degree == 0:
                g_copy.remove_node(node)
        new_num_edges = len(g_copy.edges)
        nodes, degrees = zip(*g_copy.degree)
        loop_count += 1
    random.seed(1)
    np.random.seed(1)
    nonredundant_list = []
    for edge in g_copy.edges:
        rows = pairs_df.query(f'({CLUSTER_COLUMN_1} == "{edge[0]}" and {CLUSTER_COLUMN_2} == "{edge[1]}") or ' \
                              f'({CLUSTER_COLUMN_1} == "{edge[1]}" and {CLUSTER_COLUMN_2} == "{edge[0]}")')
        if len(rows) == 1:
            row = rows.iloc[0]
        else:
            row = rows.iloc[np.random.randint(len(rows))]
        nonredundant_list.append(row)
    nonredundant_df = pd.DataFrame(nonredundant_list)
    return nonredundant_df


# If GRAND has been performed on a dataset of pairs belonging to one class
# (e.g. pairs representing protein-protein interactions), generate new pairs not belonging to the class.
# Take the cluster dictionary, dataframe of non-redundant "positive" pairs and original dataframe of all pairs.
# Return an equal number of "negative" pairs.
def get_negative_pairs(cluster_dict, positive_df, pairs_df, id_col1, id_col2):
    # Previously we needed to look up cluster membership based on ID,
    # but now we want to draw an ID from a chosen cluster, so make a new, reversed dict
    cluster_reverse_dict = {}
    for key in cluster_dict:
        val = cluster_dict[key]
        if val in cluster_reverse_dict:
            cluster_reverse_dict[val].append(key)
        else:
            cluster_reverse_dict[val] = [key]
    diffs_1 = set(pairs_df[CLUSTER_COLUMN_1]).difference(set(positive_df[CLUSTER_COLUMN_1]))
    diffs_2 = set(pairs_df[CLUSTER_COLUMN_2]).difference(set(positive_df[CLUSTER_COLUMN_2]))
    negative_pairs = []
    while len(negative_pairs) < len(positive_df):
        cls_1 = np.random.choice(list(diffs_1), 1)[0]
        cls_2 = np.random.choice(list(diffs_2), 1)[0]
        existing = pairs_df.query(f'{CLUSTER_COLUMN_1} == "{cls_1}" and {CLUSTER_COLUMN_2} == "{cls_2}" '
                                  f' or {CLUSTER_COLUMN_1} == "{cls_2}" and {CLUSTER_COLUMN_2} == "{cls_1}"')
        if len(existing) == 0:
            diffs_1.remove(cls_1)
            diffs_2.remove(cls_2)
            protein_1 = np.random.choice(cluster_reverse_dict[cls_1], 1)[0]
            protein_2 = np.random.choice(cluster_reverse_dict[cls_2], 1)[0]
            negative_pairs.append([protein_1, protein_2, cls_1, cls_2])
    negative_pairs_df = pd.DataFrame(negative_pairs,
                                     columns=[id_col1, id_col2, CLUSTER_COLUMN_1, CLUSTER_COLUMN_2])
    return negative_pairs_df
