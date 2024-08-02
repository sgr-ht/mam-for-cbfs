#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################################
#                                                                            #
#  Code for the USENIX Security '22 paper:                                   #
#  How Machine Learning Is Solving the Binary Function Similarity Problem.   #
#                                                                            #
#  MIT License                                                               #
#                                                                            #
#  Copyright (c) 2019-2022 Cisco Talos                                       #
#                                                                            #
#  Permission is hereby granted, free of charge, to any person obtaining     #
#  a copy of this software and associated documentation files (the           #
#  "Software"), to deal in the Software without restriction, including       #
#  without limitation the rights to use, copy, modify, merge, publish,       #
#  distribute, sublicense, and/or sell copies of the Software, and to        #
#  permit persons to whom the Software is furnished to do so, subject to     #
#  the following conditions:                                                 #
#                                                                            #
#  The above copyright notice and this permission notice shall be            #
#  included in all copies or substantial portions of the Software.           #
#                                                                            #
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,           #
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF        #
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND                     #
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE    #
#  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION    #
#  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION     #
#  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.           #
#                                                                            #
#  gnn_preprocessing.py - Convert each function into a graph with            #
#    BB-level features.                                                      #
#                                                                            #
##############################################################################

# some parts of this code has been changed by sgr-ht in 2023-2024

import click
import json
import networkx as nx
import numpy as np
import os

from collections import Counter
from collections import defaultdict
from scipy.sparse import coo_matrix
from tqdm import tqdm

import pickle
import pandas as pd
import fasttext



def get_top_opcodes(input_folder, num_opc):
    """
    Extract the list of most frequent opcodes across the training data.
    Args:
        input_folder: a folder with JSON files from IDA_acfg_disasm
        num_opc: the number of most frequent opcodes to select.
    Return
        dict: map most common opcodes to their ranking.
    """
    opc_cnt = Counter()

    for f_json in tqdm(os.listdir(input_folder)):
        if not f_json.endswith(".json"):
            continue

        json_path = os.path.join(input_folder, f_json)
        with open(json_path) as f_in:
            jj = json.load(f_in)

            idb_path = list(jj.keys())[0]
            # print("[D] Processing: {}".format(idb_path))
            j_data = jj[idb_path]
            del j_data['arch']

            # Iterate over each function
            for fva in j_data:
                fva_data = j_data[fva]
                # Iterate over each basic-block
                for bb in fva_data['basic_blocks']:
                    opc_cnt.update(fva_data['basic_blocks'][bb]['bb_mnems'])

    print("[D] Found: {} mnemonics.".format(len(opc_cnt.keys())))
    print("[D] Top 10 mnemonics: {}".format(opc_cnt.most_common(10)))
    return {d[0]: c for c, d in enumerate(opc_cnt.most_common(num_opc))}



def create_features_matrix_bow(node_list, fva_data, opc_dict):
    """
    Create the matrix with numerical features.
    Args:
        node_list: list of basic-blocks addresses
        fva_data: dict with features associated to a function
        opc_dict: selected opcodes.
    Return
        np.matrix: Numpy matrix with selected features.
    """
    f_mat = np.zeros((len(node_list), len(opc_dict)))

    # Iterate over each BBs
    for node_idx, node_fva in enumerate(node_list):
        if str(node_fva) not in fva_data["basic_blocks"]:
            # Skipping node
            continue
        node_data = fva_data["basic_blocks"][str(node_fva)]
        for mnem in node_data["bb_mnems"]:
            if mnem in opc_dict:
                mnem_idx = opc_dict[mnem]
                f_mat[node_idx][mnem_idx] += 1
    # WARNING
    # Forcing the type to np.int8 to limit memory usage.
    #   Use the same when parsing the data!
    return f_mat.astype(np.int8)



def create_functions_dict_bow(input_folder, opc_dict):
    """
    Convert each function into a graph with BB-level features.
    Args:
        input_folder: a folder with JSON files from IDA_acfg_disasm
        opc_dict: dictionary that maps most common opcodes to their ranking.
    Return
        dict: map each function to a graph and features matrix
    """
    try:
        functions_dict = defaultdict(dict)

        for f_json in tqdm(os.listdir(input_folder)):
            if not f_json.endswith(".json"):
                continue

            json_path = os.path.join(input_folder, f_json)
            with open(json_path) as f_in:
                jj = json.load(f_in)

                idb_path = list(jj.keys())[0]
                # print("[D] Processing: {}".format(idb_path))
                j_data = jj[idb_path]
                del j_data['arch'] 

                # Iterate over each function
                for fva in j_data:
                    fva_data = j_data[fva]
                    g_mat, nodes = create_graph(            
                        fva_data['nodes'], fva_data['edges'])
                    
                    f_mat = create_features_matrix_bow(          
                        nodes, fva_data, opc_dict)
                    
                    functions_dict[idb_path][fva] = {
                        #'graph': np_to_scipy_sparse(g_mat),
                        #'opc': np_to_scipy_sparse(f_mat)
                        'graph': g_mat,
                        'opc': f_mat
                    }

        return functions_dict

    except Exception as e:
        print("[!] Exception in create_functions_dict_bow\n{}".format(e))
        return dict()


def main_bow(input_dir, training, dimension, vocab_f_path, output_dir):

    if training:
        opc_dict = get_top_opcodes(input_dir, dimension) 
        output_path = os.path.join(output_dir, vocab_f_path)
        with open(output_path, "w") as f_out:
            json.dump(opc_dict, f_out)

    # inference
    else:
        if not os.path.isfile(vocab_f_path):
            print("[!] Error loading {}".format(vocab_f_path))
            return
        with open(vocab_f_path) as f_in:
            opc_dict = json.load(f_in)

    if not training and dimension > len(opc_dict):
        print("[!] Num opcodes is greater than training ({} > {})".format(
            dimension, len(opc_dict)))
        return

    o_dict = create_functions_dict_bow(input_dir, opc_dict)
    #o_json = "graph_func_dict_opc_{}.json".format(dimension)
    o_pkl = "graph_func_dict_opc_{}.pickle".format(dimension)
    
    #output_path = os.path.join(output_dir, o_json)
    output_path = os.path.join(output_dir, o_pkl)
    with open(output_path, 'wb') as f_out:
        #json.dump(o_dict, f_out)
        pickle.dump(o_dict, f_out)




def create_graph(nodes, edges):
    """
    Create a NetworkX direct graph from the list of nodes and edges.
    Args:
        node_list: list of nodes
        edge_list: list of edges
    Return
        np.matrix: Numpy adjacency matrix
        list: nodes in the graph
    """
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node)
    for edge in edges:
        G.add_edge(edge[0], edge[1])

    nodelist = list(G.nodes())
    adj_mat = nx.to_numpy_matrix(G, nodelist=nodelist, dtype=np.int8)
    return adj_mat, nodelist





#def np_to_scipy_sparse(np_mat):
#    """
#    Convert the Numpy matrix in input to a Scipy sparse matrix.
#    Args:
#        np_mat: a Numpy matrix
#    Return
#        str: serialized matrix
#    """
#    cmat = coo_matrix(np_mat)
#    # Custom string serialization
#    row_str = ';'.join([str(x) for x in cmat.row])
#    col_str = ';'.join([str(x) for x in cmat.col])
#    data_str = ';'.join([str(x) for x in cmat.data])
#    n_row = str(np_mat.shape[0])
#    n_col = str(np_mat.shape[1])
#    mat_str = "::".join([row_str, col_str, data_str, n_row, n_col])
#    return mat_str




def get_fastText_vocab(input_folder, dimension):
    """
    Extract the list of most frequent opcodes across the training data.
    Args:
        input_folder: a folder with JSON files from IDA_acfg_disasm
        dimension: the dimension of a BB vector
    Return
        BB dataframe for training 
    """

    list_BB_insts = list()

    for f_json in tqdm(os.listdir(input_folder)):
        if not f_json.endswith(".json"):
            continue

        #print("dataset: ",f_json)
        json_path = os.path.join(input_folder, f_json)
        with open(json_path) as f_in:
            jj = json.load(f_in)

            idb_path = list(jj.keys())[0]
            # print("[D] Processing: {}".format(idb_path))
            j_data = jj[idb_path]
            del j_data['arch']

            # Iterate over each function
            for fva in j_data:
                fva_data = j_data[fva]
                # Iterate over each basic-block
                for bb in fva_data['basic_blocks']:
                    list_BB_insts.append(fva_data['basic_blocks'][bb]['bb_mnems'])


    print("LIST: Before duplicate check: ", len(list_BB_insts))
    list_BB_insts = list(map(list, set(map(tuple, list_BB_insts))))
    #list_BB_insts = sorted(list(map(list, set(map(tuple, list_BB_insts)))))
    print("LIST: After duplicate check: ",  len(list_BB_insts))

    for bb_idx in range(len(list_BB_insts)):
      list_BB_insts[bb_idx] = ' '.join(list_BB_insts[bb_idx])


    df_BBs = pd.DataFrame(list_BB_insts, columns=['bb_mnems'])

    return df_BBs




def create_features_matrix_fastText(node_list, fva_data, dimension, fastText_model):
    """
    Create the matrix with numerical features.
    Args:
        node_list: list of basic-blocks addresses
        fva_data: dict with features associated to a function
        opc_dict: selected opcodes.
    Return
        np.matrix: Numpy matrix with selected features.
    """
    f_mat = np.zeros((len(node_list), dimension), dtype=np.float32)

    # Iterate over each BBs
    for node_idx, node_fva in enumerate(node_list):
        if str(node_fva) not in fva_data["basic_blocks"]:
            continue
        node_data = fva_data["basic_blocks"][str(node_fva)]

        if len(node_data['bb_mnems']) == 0:  
            continue

        for bb_inst in node_data['bb_mnems']:
           f_mat[node_idx] += fastText_model.get_word_vector(bb_inst) 

        f_mat[node_idx] = f_mat[node_idx] / len(node_data['bb_mnems']) 

    # WARNING
    # Forcing the type to np.int8 to limit memory usage.
    #   Use the same when parsing the data!
    # return f_mat.astype(np.int8)
    return f_mat


# fastText
def  create_functions_dict_fastText(input_folder, dimension, fastText_model):
    """
    Convert each function into a graph with BB-level features.
    Args:
        input_folder: a folder with JSON files from IDA_acfg_disasm
        dimension: the dimension of a BB vector
    Return
        dict: map each function to a graph and features matrix
    """
    try:

        functions_dict = defaultdict(dict)
        for f_json in tqdm(os.listdir(input_folder)):
            if not f_json.endswith(".json"):
                continue

            json_path = os.path.join(input_folder, f_json)
            with open(json_path) as f_in:
                jj = json.load(f_in)

                idb_path = list(jj.keys())[0]
                # print("[D] Processing: {}".format(idb_path))
                j_data = jj[idb_path]
                del j_data['arch']

                # Iterate over each function
                for fva in j_data:
                    fva_data = j_data[fva]
                    
                    g_mat, nodes = create_graph(            
                        fva_data['nodes'], fva_data['edges'])
                    
                    f_mat = create_features_matrix_fastText(          
                        nodes, fva_data, dimension, fastText_model)
                    
                    functions_dict[idb_path][fva] = {
                        #'graph': np_to_scipy_sparse(g_mat),
                        #'opc': np_to_scipy_sparse(f_mat)
                        'graph': g_mat,
                        'fastText': f_mat
                    }

        return functions_dict

    except Exception as e:
        print("[!] Exception in create_functions_dict_fastText\n{}".format(e))
        return dict()



# fastText
def main_fastText(input_dir, training, dimension, vocab_f_path, output_dir):
    print(" main_fastText.\n")   

    if training:
        print("Training mode: main_fastText.\n")
        df_fastText_train_data = get_fastText_vocab(input_dir, dimension) 

        fastText_vocab_f_name = "fastText_training_data" + "_len" + str(len(df_fastText_train_data)) + "_all_val.txt"
        output_path = os.path.join(output_dir, fastText_vocab_f_name)

        df_fastText_train_data.to_csv( output_path, index=None, columns=None, header=None)

        with open( output_path , 'r') as f_in:
            content = f_in.read()        
        content = content.replace('\"', '')
        with open(output_path, 'w') as f_out:
            f_out.write(content)


        # fastText training
        EPOCH = 200
        fastText_model = fasttext.train_unsupervised(output_path, minCount=1, epoch=EPOCH, dim=dimension, thread = 6)
        fastText_model_f_name = "fastText_model_dim" +  str(dimension)
        output_path = os.path.join(output_dir, fastText_model_f_name)
        fastText_model.save_model(output_path)

        result = fastText_model.get_nearest_neighbors('mov')
        print(result)


    # inference
    else:
        print("Inference mode: main_fastText.\n")

        if not os.path.isfile(vocab_f_path):
            print("[!] Error loading {}".format(vocab_f_path))
            return
        print("loading fastText model... :", vocab_f_path)
        fastText_model = fasttext.load_model(vocab_f_path)

        result = fastText_model.get_nearest_neighbors('mov')
        print(result)

        # dimension check
        if not training and dimension !=  int(vocab_f_path.split('_')[-1][3:]): 
            print("[!] dimension is different from training ({} != {})".format(
                dimension, int(vocab_f_path.split('_')[-1][3:]) ))
            return

        o_dict = create_functions_dict_fastText(input_dir, dimension, fastText_model)
        o_pkl = "graph_func_fastText_dim_{}.pickle".format(dimension)
    
        output_path = os.path.join(output_dir, o_pkl)
        with open(output_path, 'wb') as f_out:
            #json.dump(o_dict, f_out)
            pickle.dump(o_dict, f_out)


###
### main 
###

@click.command()
@click.option('-i', '--input-dir', required=True,
              help='IDA_acfg_disasm JSON files.')     
@click.option('--training', required=True, is_flag=True,
              help='Process training data')           
@click.option('--t_mode', required=True,
              help='Train mode: BoW, fastText')       
@click.option('-n', '--dimension',
              default=200,
              help='Number of most frequent opcodes.')
@click.option('-d', '--vocab_f_path',
              default='opcodes_dict.json',
              help='BoW: opcodes_dict.json, fastText: path to fastText model')       
@click.option('-o', '--output-dir', required=True,
              help='Output directory.')               

def main(input_dir, training, t_mode, dimension, vocab_f_path, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    if t_mode == 'BoW': 
        main_bow(input_dir, training, dimension, vocab_f_path, output_dir)

    elif t_mode == 'fastText': 
        print(" fastText mode.\n")
        main_fastText(input_dir, training, dimension, vocab_f_path, output_dir)

    else:
        print("[!] t_mode error! Please set train mode: BoW, fastText.")
        return   
    
 
if __name__ == '__main__':
   main()


