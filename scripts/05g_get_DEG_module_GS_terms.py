import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys
import argparse
import yaml
import json

# This script takes a DataFrame with subcluster DEG
# lists, and for each, gets the GO terms associated
# with each DEG for each represented cell type.
# It then produces frequency-weighted lists
# of GO terms per cell type for each DEG module.

# Script setup

# 00a. Create argparse object for reading input arguments
parser = argparse.ArgumentParser(description='Visualization of differentially expressed genes' + \
        ' for selected expression data partitions.')
parser.add_argument(f'--config-yaml-path',type=str,required=True)
args = parser.parse_args()

# 00b. Specify parameters to use 
# when parsing gene set term descriptors
# 00b.i. A list of words to ignore
WORDS_TO_IGNORE = ['to','and','for','of','by','with']
# 00b.ii. The repeat frequency threshold
# above which to assess terms containing
# the repeated words
REPEAT_FREQ_THRESHOLD = 75.0

# 01. Get analysis parameters from configuration file
cfg = None
with open(args.config_yaml_path) as f:
    cfg = yaml.safe_load(f)
# 01a. Dataset information
# 01a.i. Data root directory
qc_root_dir = cfg.get('root_dir')
# 01a.ii. Differential expression analysis output directory
dea_data_dir = qc_root_dir + f'/' + cfg.get('dea_data_dir')
dea_diffexpr_dir = dea_data_dir + f'/' + cfg.get('dea_deseq_formatted_data_dir')
dea_results_dir = dea_diffexpr_dir + f'/' + cfg.get('dea_results_dir')
dea_formatted_dir = dea_diffexpr_dir + f'/' + cfg.get('dea_ready_deseq_formatted_data_dir')
# 01a.iii. Metadata file name tag for DEA-ready metadata files
dea_fn_tag = cfg.get('dea_ready_deseq_formatted_data_fn_tag')
# 01a.iv. DEG cluster output directory
ppt_out_dir = qc_root_dir + f'/' + cfg.get('deg_clust_data_dir')
# 01a.v. Flag specifying whether to analyze
# control group instead of treatment group data
to_assess_controls = cfg.get('analyze_control_group_flag')
# 01a.vi. Tag and subdirectory for specified cell type mix
ct_mix_tag = cfg.get('cell_type_mix_tag')
# 01a.vii. Full subcluster DataFrame directory
subclust_dir = ppt_out_dir + f'/' + cfg.get('deg_sc_data_dir') + f'/{ct_mix_tag}'
if to_assess_controls:
    subclust_dir += f'/control_group'
else:
    subclust_dir += f'/treatment_group'
# 01a.x. GO term dictionary directory
go_gene_set_dir = qc_root_dir + f'/' + cfg.get('go_parent_dir') + f'/' + cfg.get('enr_dir')
# 01a.xi. GO term dictionary file
go_gene_set_fn_list = cfg.get('gseapy_enrichr_libs')
# 01a.xii. File name tag with DEA
deg_comp_out_fn_tag = cfg.get('deg_comp_tc_fn_tag')
# 01a.xiii. File name tag describing type of initial annotations
deg_subclust_fn_tag = 'test_cond_deg_subclust_w_scores'

# 01b. Read in Gene Ontology dictionaries
go_gene_set_dict = {}
go_gene_set_files = [
        _ for _ in
        os.listdir(go_gene_set_dir)
        if any(
            [
                gene_set_fn in _
                for gene_set_fn in 
                go_gene_set_fn_list
                ]
            )
        ]
if len(go_gene_set_files) > 0:
    for go_gene_set_file in go_gene_set_files:
        dict_curr = None
        go_gene_set_full_fn = f'{go_gene_set_dir}/{go_gene_set_file}'
        with open(go_gene_set_full_fn,'r') as infile:
            dict_curr = json.load(infile)
        if len(go_gene_set_dict) == 0:
            go_gene_set_dict = dict_curr
        else:
            for k,v in dict_curr.items():
                go_gene_set_dict[k] = v
else:
    go_gene_set_str = '\n\t'.join(go_gene_set_fn_list)
    err_message = f'Could not find gene set\n\t{go_gene_set_str}.\nPlease check specifications.'
    sys.exit(err_message)

# 01c. Specify a top percentile to use for 
# finding top gene sets
top_pctile = 95.0

# 02. Read in DataFrame with DEG subcluster/"module" information
subclust_files = [
        _ for _ in
        os.listdir(subclust_dir)
        if all(
            [
                tag in _
                for tag in
                [
                    deg_comp_out_fn_tag,
                    deg_subclust_fn_tag,
                    '.csv'
                    ]
                ]
            )
        ]
subclust_df = pd.DataFrame()
if len(subclust_files) > 0:
    subclust_file = subclust_files[0]
    print(f'Reading DEG subcluster information from\n\t{subclust_file}...')
    subclust_full_fn = f'{subclust_dir}/{subclust_file}'
    subclust_df = pd.read_csv(
            subclust_full_fn,
            index_col=0
            )

# 03. For each DEG subcluster, pull the DEGs, split them by 
# cell type, and get the associated GO terms for each cell type
subclust_out_fn_dict = {}
all_cell_types = []
print(f'Getting GO gene set annotations for DEG subclusters....')
for row_idx in subclust_df.index.values.tolist():
    print(f'\trow {row_idx+1} of {len(subclust_df)}')
    row = subclust_df.loc[row_idx]
    # 03a. Set up a dictionary to 
    # hold GO gene sets and weights for each subcluster
    curr_row_dict = {}
    test_cond = row['test_cond']
    clust_num = row['clust_num']
    clust_color = row['clust_color']
    subclust_num = row['subclust']
    subclust_sim_score = row['mean_similarity']
    
    # 03b. Set up an output directory
    # for plots related to the current test condition
    out_dir_curr = f'{subclust_dir}/{test_cond}_subclust_annotations'
    if not os.path.isdir(out_dir_curr):
        os.system(f'mkdir -p {out_dir_curr}')
    # 03b.i. Set up the output file name
    # for the current subcluster's ranked
    # gene sets plot (when that option is enabled)
    deg_ann_plot_fn = f'{test_cond}__clust_{clust_num}_subclust_{subclust_num}.png'
    # 03b.ii. Add the test condition
    # to the dictionary of Excel writers
    # if it does not already exist
    if test_cond not in subclust_out_fn_dict.keys():
        test_cond_out_fn = f'{out_dir_curr}/{test_cond}__cell_type_deg_gene_set_enrich_by_subclust.txt'
        if os.path.isfile(test_cond_out_fn):
            print(f'An old copy of\n\t{test_cond_out_fn}\nexists; deleting it now')
            os.system(f'rm {test_cond_out_fn}')
        subclust_out_fn_dict[test_cond] = test_cond_out_fn

    # 03c. Get the DEGs in the current
    # cluster and find the unique cell
    # types they represent
    degs = row['degs_subclust'].split(',')
    deg_pieces = [
            _.split('_')
            for _ in degs
            ]
    cts = [
            _[0] for _ in 
            deg_pieces
            ]
    ct_unique = list(set(cts))
    all_cell_types.extend(ct_unique)

    # 03e. Build the list of cell-type-specific
    # gene sets enriched for the current subcluster
    # 03e.i. Build an output string for writing results
    # to a text file for visual inspection
    out_txt_str_curr = f'Cluster {clust_num} ({clust_color}), Subcluster {subclust_num}:\n'
    out_txt_str_curr += f'Similarity score: {subclust_sim_score:0.3}\n'
    for ax_idx_curr,ct_curr in enumerate(ct_unique):
        deg_idxs = [
                i for i,_ in
                enumerate(deg_pieces)
                if _[0]==ct_curr
                ]
        degs_ct_curr = [
                deg_pieces[i][2]
                for i in deg_idxs
                ]
        degs_ct_curr_str = '\n\t'.join(degs_ct_curr)
        out_txt_str_curr += f'{ct_curr} DEGs:\n\t{degs_ct_curr_str}\n'
        subclust_df.at[row_idx,f'n_degs_{ct_curr}'] = len(degs_ct_curr)
        subclust_df.at[row_idx,f'degs_{ct_curr}'] = ';'.join(degs_ct_curr)

        # 03b. Build GO-based dictionaries of terms
        # for the DEGs of the current cell type
        go_terms_ct_curr = {}
        msigdb_terms_ct_curr = {}
        for deg_ct_curr in degs_ct_curr:
            go_terms_deg_ct_curr = [
                    k for k,v in 
                    go_gene_set_dict.items()
                    if deg_ct_curr in v
                    ]
            go_terms_ct_curr[deg_ct_curr] = go_terms_deg_ct_curr

        # 03c. Check whether any terms are recurring
        # 03c.i. GO terms
        all_go_terms_ct_curr = [
                _
                for item in go_terms_ct_curr.values()
                for _ in item
                ]
        n_degs_got_curr = len(go_terms_ct_curr.values())
        got_u,n_got_u = np.unique(
                all_go_terms_ct_curr,
                return_counts=True
                )
        frac_got_u = [
                _/n_degs_got_curr
                for _ in n_got_u
                ]

        # 03d. Get top specified percentile
        if len(n_got_u) > 0:
            got_top_pctile_val = np.percentile(
                    n_got_u,
                    top_pctile
                    )
            got_tpc_idxs = [
                    i for i,_ in
                    enumerate(n_got_u)
                    if _ >= got_top_pctile_val
                    ]
            got_tpc_u = [
                    got_u[i] for i in got_tpc_idxs
                    ]
            got_tpc_counts = [
                    n_got_u[i] for i in got_tpc_idxs
                    ]
            got_tpc_fracs = [
                    frac_got_u[i] for i in got_tpc_idxs
                    ]
            # 03d.i. Sort values by descending number
            # of hits for easier visualization of results
            got_tpc_idxs_desc = [
                    _ for _ in
                    reversed(
                        np.argsort(got_tpc_counts)
                        )
                    ]
            got_tpc_u_desc = [
                    got_tpc_u[i] for i in got_tpc_idxs_desc
                    ]
            got_tpc_counts_desc = [
                    got_tpc_counts[i] for i in got_tpc_idxs_desc
                    ]
            got_tpc_fracs_desc = [
                    got_tpc_fracs[i] for i in got_tpc_idxs_desc
                    ]
            # 03d.i.A. Look for repeats in the GO top-percentile terms
            # Note: GO terms are written with a sentence-like
            # formatting, with a description followed by a GO
            # term ID in parentheses. We take the words in the
            # description only
            got_pieces = [
                    _.split('(')[0].strip().split(' ')
                    for _ in got_tpc_u_desc
                    ]
            got_words = [
                    item for _ in
                    got_pieces
                    for item in _
                    ]
            got_words = [
                    _ for _ in
                    got_words
                    if _ not in
                    WORDS_TO_IGNORE
                    ]
            got_words_u, n_rep_got_words = np.unique(
                    got_words,
                    return_counts=True
                    )
            # 03d.i.B. Get the words from the upper quartile in repeat frequencies
            got_idx_d = [
                    _ for _ in
                    reversed(
                        np.argsort(n_rep_got_words)
                        )
                    ]
            got_wu_d = [
                    got_words_u[i] for i in got_idx_d
                    ]
            n_got_wu_d = [
                    n_rep_got_words[i] for i in got_idx_d
                    ]
            got_rep_freq_thresh = np.percentile(
                    n_got_wu_d,
                    REPEAT_FREQ_THRESHOLD
                    )
            got_wu_above_rft = [
                    got_wu_d[i]
                    for i,_ in enumerate(n_got_wu_d)
                    if _>got_rep_freq_thresh
                    ]
            got_repeated_terms = [
                    _ for _ in
                    got_tpc_u_desc
                    if any(
                        [
                            tag in _
                            for tag in
                            got_wu_above_rft
                            ]
                        )
                    ]
            subclust_df.at[row_idx,f'{ct_curr}_top_{top_pctile}th_pctile_val_got'] = got_top_pctile_val
            subclust_df.at[row_idx,f'{ct_curr}_top_{top_pctile}th_pctile_go_terms'] = ';'.join(got_tpc_u_desc)
            subclust_df.at[row_idx,f'{ct_curr}_top_{top_pctile}th_pctile_go_term_counts'] = ';'.join([f'{_}' for _ in got_tpc_counts_desc])
            subclust_df.at[row_idx,f'{ct_curr}_top_{top_pctile}th_pctile_go_term_fracs'] = ';'.join([f'{_:0.2}' for _ in got_tpc_fracs_desc])
            subclust_df.at[row_idx,f'{ct_curr}_top_{top_pctile}th_pctile_got_top_quartile_repeated_words'] = ';'.join(got_wu_above_rft)
            subclust_df.at[row_idx,f'{ct_curr}_top_{top_pctile}th_pctile_got_terms_w_top_words'] = ';'.join(got_repeated_terms)
            tpgt_str = '\n\t'.join(
                    [
                        f'{gt_frac:0.2} ({gt_count}/{n_degs_got_curr}):\t\t{gt_name}'
                        for gt_frac,gt_count,gt_name in
                        zip(
                            got_tpc_fracs_desc,
                            got_tpc_counts_desc,
                            got_tpc_u_desc
                            )
                        ]
                    )
            out_txt_str_curr += f'{ct_curr} top GO gene set hits (frac DEGs hit: name):\n\t{tpgt_str}\n'

        # 03d. Add info to DataFrame
        subclust_df.at[row_idx,f'{ct_curr}_unique_go_terms'] = ';'.join(got_u)
        subclust_df.at[row_idx,f'{ct_curr}_unique_go_term_counts'] = ';'.join([str(_) for _ in n_got_u])

        # 03e. Write information for the current subcluster
        # to a text file for the appropriate test
        # condition for visual inspection
        out_txt_str_curr += f'\n\n'
        curr_fn = subclust_out_fn_dict[test_cond]
        with open(curr_fn,'a') as f:
            f.write(out_txt_str_curr)

# 05. Save a skeleton copy of the
# output DataFrame for the results
# of manual review, with main repeated keywords
all_cell_types = sorted(
        list(
            set(
                all_cell_types
                )
            )
        )
cols_skeleton = [
        'test_cond',
        'clust_color',
        'clust_num',
        'subclust',
        'mean_similarity'
        ]
subclust_df_skel = subclust_df[cols_skeleton].copy()
ann_col_tags = [
        'GO_summary',
        'summary'
        ]
for ann_col_tag in ann_col_tags:
    for ct_curr in all_cell_types:
        subclust_df_skel[f'{ct_curr}_{ann_col_tag}'] = None
subclust_skel_out_fn = f'{subclust_dir}/{deg_comp_out_fn_tag}__all_subclust_sim_score_go_ann__for_review.csv'
subclust_df_skel.to_csv(
        subclust_skel_out_fn,
        index=True
        )
sys.exit()







