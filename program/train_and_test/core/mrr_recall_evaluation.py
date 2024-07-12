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
#  MRR@10 and Recall@K                                                       #
#                                                                            #
##############################################################################

# some parts of this code has been changed by sgr-ht in 2023-2024

import math
import numpy as np
import os
import pandas as pd
import scipy.stats as stats
import shutil

from collections import defaultdict
from sklearn import metrics



def merge_data(df_pairs, df_similarity):
    df_pairs = df_pairs.merge(
        df_similarity,
        how='left',
        #left_on=['idb_path_1', 'fva_1', 'idb_path_2', 'fva_2'],  
        #right_on=['idb_path_1', 'fva_1', 'idb_path_2', 'fva_2'])  
        left_on=['idb_path_1', 'fva_1', 'idb_path_2', 'fva_2', 'db_type'],
        right_on=['idb_path_1', 'fva_1', 'idb_path_2', 'fva_2','db_type'])
        
    return df_pairs



def compute_ranking(df_pos, df_neg, test_name, r_dict, rank_method):
    NUM_NEGATIVES = 100
    
    TH_LIST = [1] + list(range(5, 15, 5)) 
    for task in sorted(set(df_pos['db_type'])):
        df_pos_task = df_pos[df_pos['db_type'] == task]
        df_neg_task = df_neg[df_neg['db_type'] == task]

        tot_pos = df_pos_task.shape[0] 

        # Compute the ranking for all the positive test cases
        rank_list = list()
        for idx, group in df_neg_task.groupby(['idb_path_1', 'fva_1']):
            c1 = (df_pos_task['idb_path_1'] == idx[0])
            c2 = (df_pos_task['fva_1'] == idx[1])
            pos_pred = df_pos_task[c1 & c2]['sim'].values[0]
            neg_pred = list(group['sim'].values)

            assert(len(neg_pred) == NUM_NEGATIVES)
            ranks = stats.rankdata([pos_pred] + neg_pred, method=rank_method) 
            rank_list.append(NUM_NEGATIVES + 2 - ranks[0])

        # Compute the ranking list
        cc_list = list()
        for th in TH_LIST:
            cc_list.append(len([x for x in rank_list if x <= th])) 

        # MRR@10 metric
        tmp_list = [1 / x if x <= 10 else 0 for x in rank_list] 
        MRR = sum(tmp_list) / len(tmp_list)

        # Save data in a temporary dictionary.
        if task not in r_dict:
            r_dict[task] = defaultdict(list)

        r_dict[task]['model_name'].append(test_name)
        for th, cc in zip(TH_LIST, cc_list):
            r_dict[task]["Recall@{}".format(th)].append(cc / tot_pos)
        r_dict[task]["MRR@10"].append(MRR)



def compute_mrr_and_recall(df_pos, df_neg, df_pos_sim, df_neg_sim):
    # Alternatives: min or max
    rank_method = 'max'

    results_dict = dict()

    assert(df_pos_sim.isna().sum()['sim'] == 0)
    assert(df_neg_sim.isna().sum()['sim'] == 0)

    del df_pos_sim['func_name_1']
    del df_pos_sim['func_name_2']

    del df_neg_sim['func_name_1']
    del df_neg_sim['func_name_2']
    df_pos_m = merge_data(df_pos, df_pos_sim) 
    df_neg_m = merge_data(df_neg, df_neg_sim) 

    test_name = "val_mrr_recall"
    
    compute_ranking(df_pos_m, df_neg_m, test_name,
                    results_dict, rank_method=rank_method)

    
    return results_dict["XM"]["MRR@10"][0], results_dict["XM"]["Recall@1"][0], \
           results_dict["XM"]["Recall@5"][0], results_dict["XM"]["Recall@10"][0], results_dict

