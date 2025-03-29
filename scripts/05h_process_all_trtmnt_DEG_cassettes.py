import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys
import argparse
import yaml
import json
import networkx as nx
import obonet as ob

# This script processes all previously identified
# DEG subclusters for a DEA, retains only subclusters
# with DEGs of mixed cell types, and
# (1) filters them to remove subclusters
#     with exceptionally low average similarity,
#     such that only highly coordinated subclusters
#     are retained for downstream analysis
# (2) generates several annotations to facilitate
#     functional analysis:
#     (i) assigns DEGs within a subcluster to "blocks"
#         with the same cell type and direction of dysregulation
#         (e.g. "MG_up") for inspection
#     (ii) adds Gene Ontology (GO) terms for each DEG
#          where possible

# Script setup

# 00a. Create argparse object for reading input arguments
parser = argparse.ArgumentParser(description='Visualization of differentially expressed genes' + \
        ' for selected expression data partitions.')
parser.add_argument(f'--config-yaml-path',type=str,required=True)
args = parser.parse_args()

# 00b. Specify a significance level for statistical tests
ALPHA = 0.05

# 00c. Get list of named colors
color_list = mcolors.CSS4_COLORS
plt_color_dict = {
        'hiv_sud_status_pos_u_y_pos_u_n':'darkviolet',
        'hiv_sud_status_pos_d_y_pos_d_n':'magenta',
        'hiv_sud_status_neg_y_neg_n':'dodgerblue',
        }

# 00d. Set matplotlib font size
plt.rcParams.update({'font.size': 6})

# 00e. Specify a tag for GO terms indicating whether
# the term relates to a response or regulation
response_info_tag = 'Response To'
pos_reg_info_tag = 'Positive Regulation Of'
neg_reg_info_tag = 'Negative Regulation Of'

# 00f. Specify a test condition mapping such that
# highly-coordinated DEG cassettes in the key
# test condition are checked for overlaps in
# the highly-coordinated and full DEG lists
# of the value test condition
deg_overlap_check_tc_map = {
        'hiv_sud_status_neg_y_neg_n' : 'hiv_sud_status_pos_u_y_pos_u_n',
        'hiv_sud_status_pos_u_y_pos_u_n' : 'hiv_sud_status_neg_y_neg_n',
        'hiv_sud_status_pos_u_y_neg_y' : 'hiv_sud_status_pos_u_n_neg_n',
        'hiv_sud_status_pos_u_n_neg_n' : 'hiv_sud_status_pos_u_y_neg_y'
        }

# 01. Get analysis parameters from configuration file
cfg = None
with open(args.config_yaml_path) as f:
    cfg = yaml.safe_load(f)
# 01a. Dataset information
# 01a.i. Root directory
qc_root_dir = cfg.get('root_dir')
# 01a.ii. DEG cluster directory
deg_clust_dir = qc_root_dir + f'/' + cfg.get('deg_clust_data_dir')
# 01a.iii. DEG subcluster directory
deg_sc_dir = deg_clust_dir + f'/' + cfg.get('deg_sc_data_dir')
# 01a.iv. Cell type mix tag (for directory)
cell_type_mix_tag = cfg.get('cell_type_mix_tag')
if cell_type_mix_tag is not None:
    if cell_type_mix_tag != '':
        deg_sc_dir += f'/{cell_type_mix_tag}'

# 01a.iv. Dictionary specifying paths to DEA
# DEG subclusters of interest
deg_cassette_fn_dict = cfg.get('ann_deg_cassete_fn_dict')
# 01a.v. GO term and ontology file parent directory
go_parent_dir = cfg.get('go_parent_dir')
# 01a.iv. GO term dictionary directory
go_gene_set_dir = go_parent_dir + f'/' + cfg.get('enr_dir')
# 01a.v. GO term dictionary file name list
go_gene_set_fn_list = cfg.get('gseapy_enrichr_libs')
# 01a.vi. GO term ontology file name
go_ont_fn = cfg.get('go_obo_fn')
# 01a.vii. List of cell types in the
# current DEG subcluster analysis
ct_list = cfg.get('ct_list')
# 01a.viii. DEG subcluster mean similarity
# distribution percentile for removing
# low-similarity DEG subclusters
SIM_PCTILE = cfg.get('lower_threshold_dsc_sim')

# 01b. Read in specified Gene Ontology dictionaries
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

# 01c. Read in Gene Ontology file as a networkx
# Multi Digraph to help iwth GO term interpretations
go_ont_full_fn = qc_root_dir + f'/' + go_parent_dir + f'/' + go_ont_fn
go_graph = None
with open(go_ont_full_fn,'r') as f:
    go_graph = ob.read_obo(f)
# 01c.i. Make a mapping from GO term ID
# to the GO term name/description
id_to_name = {id_: data.get('name') for id_, data in go_graph.nodes(data=True)}

# 02. Generate DEG subcluster annotations
# 02a. Initialize DEG subcluster DataFrame
deg_sc_info_df = pd.DataFrame()

# 02b. Process treatment group data only
for group_level,info in deg_cassette_fn_dict.items():
    if group_level=='treatment':
        gl_dir = deg_sc_dir + f'/' + info['dir']
        gl_tc_info = info['test_cond_sets']
        for tc_set,tc_set_info in gl_tc_info.items():
            deg_sc_info_fn = tc_set_info['deg_info_fn']
            deg_sc_info_full_fn = gl_dir + f'/' + deg_sc_info_fn
            deg_sc_info_df_curr = pd.read_csv(
                    deg_sc_info_full_fn,
                    index_col=0
                    )
            deg_sc_info_df_curr['group_level'] = [group_level]*len(deg_sc_info_df_curr)
            if len(deg_sc_info_df) == 0:
                deg_sc_info_df = deg_sc_info_df_curr.copy()
            else:
                deg_sc_info_df = pd.concat(
                        [
                            deg_sc_info_df,
                            deg_sc_info_df_curr
                            ],
                        ignore_index=True
                        )

# 03. Get the subcluster mean similarity distribution per DEA,
# plot them, and get the similarity score threshold below which
# to remove cassettes from downstream analysis
sim_vals = deg_sc_info_df['mean_similarity'].values.tolist()
# 03b. Plot similarity scores for each individual DEA for comparison
# vs. the distribution for all DEAs combined
binedges = np.arange(
        0,
        np.nanmax(sim_vals),
        0.025
        )
plt.figure()
plt.hist(
        sim_vals,
        bins=binedges,
        alpha=0.6
        )
for pctl in np.arange(5,25,5):
    pctl_val = np.nanpercentile(
            sim_vals,
            pctl
            )
    plt.plot(
            [pctl_val]*2,
            [0,50],
            'g',
            alpha=0.6
            )
    print(f'{pctl}th percentile value: {pctl_val}')
plt.xlabel('Subcluster Similarity Score')
plt.ylabel('Frequency')
sim_plot_fn = deg_sc_dir + f'/treatment_groups_sim_score_dist.png'
plt.savefig(
        sim_plot_fn,
        dpi=300
        )
plt.close('all')

sim_pctiles_tc = []
n_subplot_rows = len(deg_sc_info_df.test_cond.unique())
fig,axs = plt.subplots(
        nrows=n_subplot_rows,
        ncols=1,
        sharex=True
        )
all_degs_by_tc_dict = {}
# 03b.i. Set up plot color dict keys for plotting
# for the currently available test conditions
plt_color_dict_keys_curr = [
        k for k in
        plt_color_dict.keys()
        if k in
        deg_sc_info_df.test_cond.unique().tolist()
        ]
for test_cond, tcdf in deg_sc_info_df.groupby('test_cond'):
    sims_curr = tcdf['mean_similarity'].values.tolist()
    # 03b.i. Build a list of all DEGs
    # for each test condition for downstream
    # highly-coordinated DEG overlap assessment
    degs_curr = [
            _.split(',')
            for _ in
            tcdf['degs_subclust'].values.tolist()
            ]
    degs_curr = [
            item
            for _ in
            degs_curr
            for item in _
            ]
    degs_curr = [
            _.replace('_up_','_').replace('_down_','_')
            for _ in
            degs_curr
            ]
    all_degs_by_tc_dict[test_cond] = degs_curr
    subplot_counter = [
            i for i,_ in
            enumerate(
                [
                    k
                    for k in
                    plt_color_dict_keys_curr
                    ]
                )
            if _==test_cond
            ][0]
    print(f'{test_cond}: {len(sims_curr)} DEG cassettes')
    sim_pctile_curr = np.nanpercentile(
            sims_curr,
            SIM_PCTILE
            )
    ax_curr = None
    if n_subplot_rows>1:
        ax_curr = axs[subplot_counter]
    else:
        ax_curr = axs

    sim_pctiles_tc.append(sim_pctile_curr)
    ax_curr.hist(
            sims_curr,
            bins=binedges,
            color=plt_color_dict[test_cond],
            alpha=0.6
            )
    ax_curr.plot(
            [sim_pctile_curr]*2,
            [0,20],
            color=plt_color_dict[test_cond],
            alpha=0.6,
            linewidth=1
            )
    ax_curr.set_title(
            test_cond,
            fontsize=4
            )
plt.xlabel(
        'Subcluster Similarity Score',
        fontsize=6)
fig.supylabel(
        'Frequency',
        fontsize=6
        )
fig.tight_layout()
tc_sim_plot_fn = deg_sc_dir + f'/treatment_groups_sim_score_dists.png'
plt.savefig(
        tc_sim_plot_fn,
        dpi=300
        )

# 03c. Choose the similarity threshold to be the point of the
# curve in the combined similarity score distribution
# Note: with the current dataset, the elbow was found using
# plots of various percentiles and occurred at just about the
# 15th percentile, at 0.752
sim_score_threshold = 0.75

# 03d. Get the cassettes with significant similarity scores
deg_sc_highly_coord_df = deg_sc_info_df.loc[
        deg_sc_info_df['mean_similarity'] >= sim_score_threshold
        ].copy()
# 03d.i. Save the DataFrame with highly
# coordinated DEG cassette information
deg_sc_highly_coord_fn = deg_sc_dir + f'/highly_coordinated_DEG_cassette_info.csv'
deg_sc_highly_coord_df.to_csv(
        deg_sc_highly_coord_fn,
        index=True
        )

# 04. Get cassettes with mixed cell types
for r_idx in deg_sc_highly_coord_df.index.values.tolist():
    row = deg_sc_highly_coord_df.loc[r_idx]
    cts = list(
            np.unique(
                [
                    _.split('_')[0] + '_'
                    for _ in 
                    row['degs_subclust'].split(',')
                    ]
                )
            )
    deg_sc_highly_coord_df.at[r_idx,'ct'] = ','.join(cts)
    if set(cts)==set(ct_list):
        deg_sc_highly_coord_df.at[r_idx,'mixed_ct'] = True
    else:
        deg_sc_highly_coord_df.at[r_idx,'mixed_ct'] = False

mixed_ct_hc_df = deg_sc_highly_coord_df.loc[
        deg_sc_highly_coord_df['mixed_ct']==True
        ].copy()
mixed_ct_hc_fn = deg_sc_dir + f'/highly_coordinated_mixed_cell_type_DEG_cassette_info.csv'
mixed_ct_hc_df.to_csv(
        mixed_ct_hc_fn,
        index=True
        )
# 04a. Pull the highly-coordinated DEG lists
# per test condition in the same way used for the
# all-DEG DataFrame
hc_mct_degs_by_tc_dict = {}
for test_cond, hcmctcdf in mixed_ct_hc_df.groupby('test_cond'):
    degs_hcmct_curr = [
            _.split(',')
            for _ in
            hcmctcdf['degs_subclust'].values.tolist()
            ]
    degs_hcmct_curr = [
            item
            for _ in
            degs_hcmct_curr
            for item in _
            ]
    degs_hcmct_curr = [
            _.replace('_up_','_').replace('_down_','_')
            for _ in
            degs_hcmct_curr
            ]
    hc_mct_degs_by_tc_dict[test_cond] = degs_hcmct_curr

# 05. Break mixed-cell-type DEG cassettes into sub-blocks based
# on their cell-type-dysregulation-direction information, and
# get the top GO terms from each sub-block.
all_go_terms_deg_sbs_by_deg_dict = {}
all_go_terms_compressed_deg_sbs_by_deg_dict = {}
all_go_terms_pot_obsolete_deg_sbs_by_deg_dict = {}
all_go_terms_deg_sbs_comb_list_dict = {}
all_block_num_to_deg_map_dict = {}
for r_idx in mixed_ct_hc_df.index.values.tolist():
    row = mixed_ct_hc_df.loc[r_idx]
    degs_curr = row['degs_subclust'].split(',')
    degs_check_str = '\n\t'.join(degs_curr)
    print(f'Row {r_idx} DEGs:\n\t{degs_check_str}')
    # 05a. Build blocks with the same DEG name tag
    deg_tags_curr = [
            '_'.join(
                _.split('_')[:-1]
                )
            for _ in
            degs_curr
            ]
    deg_names_curr = [
            _.split('_')[-1]
            for _ in
            degs_curr
            ]
    deg_ct_names_curr = [
            _.replace('_up_','_').replace('_down','_')
            for _ in
            degs_curr
            ]
    id_start_block_curr = 0
    id_blocks = []
    id_block_curr = []
    while id_start_block_curr < len(deg_tags_curr):
        if len(id_block_curr)==0:
            id_block_curr.append(id_start_block_curr)
            id_start_block_curr += 1
        elif (deg_tags_curr[id_start_block_curr]==deg_tags_curr[id_block_curr[0]]):
            id_block_curr.append(id_start_block_curr)
            id_start_block_curr += 1
        else:
            id_blocks.append(id_block_curr)
            id_block_curr = [id_start_block_curr]
            id_start_block_curr += 1
    # 05a.i. Append final ID block to the list
    id_blocks.append(id_block_curr)
    # 05a.ii. Check: print blocks
    #print(f'Blocks: {id_blocks}\n')

    # 05b. Iterate through blocks and get GO terms for each
    go_terms_deg_sbs_by_deg_dict = {}
    go_terms_compressed_deg_sbs_by_deg_dict = {}
    go_terms_pot_obsolete_deg_sbs_by_deg_dict = {}
    go_terms_deg_sbs_comb_list_dict = {}
    block_num_to_deg_map_dict = {}
    for block_num, id_block_curr in enumerate(id_blocks):
        degs_block_curr = [
                degs_curr[_]
                for _ in
                id_block_curr
                ]
        deg_names_block_curr = [
                deg_names_curr[_]
                for _ in
                id_block_curr
                ]
        block_go_dict = {}
        block_compressed_go_dict = {}
        block_pot_obsolete_go_dict = {}
        all_go_list = []
        for deg_block_curr,deg_name_block_curr in zip(degs_block_curr,deg_names_block_curr):
            go_terms_deg_sb_curr = [
                    k for k,v in
                    go_gene_set_dict.items()
                    if deg_name_block_curr in v
                    ]
            # 05b.i. Attempt to collapse the terms for the current
            # DEG in the block, by finding the most specific terms
            # from each branch of the tree
            go_ids_deg_sb_curr = [
                    _.split('(')[-1].split(')')[0]
                    for _ in go_terms_deg_sb_curr
                    ]
            # 05b.i.A. Iterate through the GO terms (using their GO
            # IDs) and remove any ancestors for the current GO term
            # from the list of GO terms, since these are more gen-
            # eralized 'parent' annotations
            go_ids_compressed_deg_sb_curr = go_ids_deg_sb_curr
            go_idx_pot_obsolete = []
            for go_id_curr in go_ids_deg_sb_curr:
                # 05b.i.B. Check the the node is in the GO
                # ontology graph; if not, since it is in the
                # JSON file with of GO terms and annotations,
                # we may guess that it has been marked as obsolete,
                # so that all relations to it in the graph have been
                # removed.
                if go_id_curr in go_graph.nodes():
                    # 05b.i.C. If the GO ID is in the
                    # graph, find its GO ancestors
                    # and remove them from our list
                    # of GO term IDs
                    ancestor_go_ids = nx.ancestors(
                            go_graph,
                            go_id_curr
                            )
                    go_ids_compressed_deg_sb_curr = [
                            _ for _ in
                            go_ids_compressed_deg_sb_curr
                            if _ not in ancestor_go_ids
                            ]
                else:
                    # 05b.i.D. If the GO ID is not in the
                    # graph, add it to the list of potentially
                    # obsolete IDs and remove it from the
                    # compressed list of GO term IDs
                    go_idx_pot_obsolete.append(go_id_curr)
                    go_ids_compressed_deg_sb_curr = [
                            _ for _ in 
                            go_ids_compressed_deg_sb_curr
                            if _ != go_id_curr
                            ]
            # 05b.ii. Convert the lists of compressed and potentially
            # obsolete GO IDs back to full GO terms
            go_terms_compressed_deg_sb_curr = [
                    go_terms_deg_sb_curr[i]
                    for i,_ in
                    enumerate(go_ids_deg_sb_curr)
                    if _ in
                    go_ids_compressed_deg_sb_curr
                    ]
            go_terms_pot_obsolete_deg_sb_curr = [
                    go_terms_deg_sb_curr[i]
                    for i,_ in
                    enumerate(go_ids_deg_sb_curr)
                    if _ in
                    go_idx_pot_obsolete
                    ]
            block_go_dict[deg_block_curr] = go_terms_deg_sb_curr
            block_compressed_go_dict[deg_block_curr] = go_terms_compressed_deg_sb_curr
            block_pot_obsolete_go_dict[deg_block_curr] = go_terms_pot_obsolete_deg_sb_curr
            all_go_list.extend(go_terms_deg_sb_curr)
            # 05b.iii. Delete current GO term lists
            # to avoid accidental filling of GO term
            # sets with previous data
            del go_terms_deg_sb_curr, go_ids_deg_sb_curr
            del go_terms_compressed_deg_sb_curr, go_terms_pot_obsolete_deg_sb_curr
        go_terms_deg_sbs_by_deg_dict[block_num] = block_go_dict
        go_terms_compressed_deg_sbs_by_deg_dict[block_num] = block_compressed_go_dict 
        go_terms_pot_obsolete_deg_sbs_by_deg_dict[block_num] = block_pot_obsolete_go_dict
        go_terms_deg_sbs_comb_list_dict[block_num] = all_go_list
        block_num_to_deg_map_dict[block_num] = degs_block_curr
    all_go_terms_deg_sbs_by_deg_dict[r_idx] = go_terms_deg_sbs_by_deg_dict
    all_go_terms_compressed_deg_sbs_by_deg_dict[r_idx] = go_terms_compressed_deg_sbs_by_deg_dict 
    all_go_terms_pot_obsolete_deg_sbs_by_deg_dict[r_idx] = go_terms_pot_obsolete_deg_sbs_by_deg_dict
    all_go_terms_deg_sbs_comb_list_dict[r_idx] = go_terms_deg_sbs_comb_list_dict
    all_block_num_to_deg_map_dict[r_idx] = block_num_to_deg_map_dict

# 05c. Consistency check: run through each DEG cassette's
# data, find the full list of GO terms, the compressed list,
# and the potentially obselete list, and print them out
out_check_str = f'\nChecking subblock GO term parsing...'
#print(f'\nChecking subblock GO term parsing...')
for c_idx in all_go_terms_deg_sbs_by_deg_dict.keys():
    out_check_str += f'\n\tDEG cassette ID {c_idx}:'
    #print(f'\tDEG cassette ID {c_idx}:')
    all_degs_subblocks = all_go_terms_deg_sbs_by_deg_dict[c_idx]
    comp_degs_subblocks = all_go_terms_compressed_deg_sbs_by_deg_dict[c_idx]
    pot_obs_degs_subblocks = all_go_terms_pot_obsolete_deg_sbs_by_deg_dict[c_idx]
    for subblock_key, subblock_all_degs in all_degs_subblocks.items():
        out_check_str += f'\n\t\tSubblock {subblock_key}:'
        #print(f'\t\tSubblock {subblock_key}:')
        subblock_comp_degs = comp_degs_subblocks[subblock_key]
        subblock_pot_obs_degs = pot_obs_degs_subblocks[subblock_key]
        for subblock_deg, subblock_go_terms_all in subblock_all_degs.items():
            n_all = len(subblock_go_terms_all)
            out_check_str += f'\n\t\t{subblock_deg}:\n\t\t\tAll GO terms ({n_all}):\n\t\t\t'
            #print(f'\t\t{subblock_deg}:\n\t\t\tAll GO terms:')
            sb_ad_str = '\n\t\t\t'.join(subblock_go_terms_all)
            out_check_str += sb_ad_str
            #print(sb_ad_str)
            out_check_str += f'\n'
            out_check_str += f'\n\t\t\tCompressed GO terms'
            #print(f'\n\t\t\tCompressed GO terms:')
            if subblock_deg in subblock_comp_degs.keys():
                len_c = len(subblock_comp_degs[subblock_deg])
                if len_c > 0:
                    out_check_str += f' ({len_c}):\n\t\t\t'
                    sb_cd_str = '\n\t\t\t'.join(
                            subblock_comp_degs[subblock_deg]
                            )
                    #print(sb_cd_str)
                    out_check_str += sb_cd_str
                else:
                    out_check_str += f':\n\t\t\tNo GO terms left after compression\n'
                    #print(f'\n\t\t\tNo GO terms left after compression')
            else:
                out_check_str += f':\n\t\t\tNo GO terms left after compression\n'
                #print(f'\n\t\t\tNo GO terms left after compression')
            out_check_str += f'\n\n\t\t\tPotentially Obsolete GO terms'
            if subblock_deg in subblock_pot_obs_degs.keys():
                len_p = len(subblock_pot_obs_degs[subblock_deg])
                if len_p > 0:
                    out_check_str += f' ({len_p}):\n\t\t\t'
                    sb_ob_str = '\n\t\t\t'.join(
                            subblock_pot_obs_degs[subblock_deg]
                            )
                    out_check_str += sb_ob_str
                    #print(sb_ob_str)
                else:
                    out_check_str += f':\n\t\t\tNo potentially obsolete GO terms found\n'
                    #print(f'\n\t\t\tNo potentially obsolete GO terms found')
            else:
                out_check_str += f':\n\t\t\tNo potentially obsolete GO terms found\n'
                #print(f'\n\t\t\tNo potentially obsolete GO terms found')
out_check_fn = deg_sc_dir + f'/mixed_cell_type_DEG_cassette_GO_term_processing_check.txt'
with open(out_check_fn,'w') as f:
    f.writelines(out_check_str)

# 06. Take the compressed list of GO terms per DEG and attempt to make sub-block
# annotations out of them
out_dir = deg_sc_dir + f'/mixed_cell_type__high_coord__subblocked_DEG_sc_anns'
if not os.path.isdir(out_dir):
    os.system(f'mkdir -p {out_dir}')
for c_idx, cassette_info in all_go_terms_compressed_deg_sbs_by_deg_dict.items():
    print(f'Processing compressed annotations for mixed-cell-type DEG cassette {c_idx}...')
    subclust_info = mixed_ct_hc_df.loc[c_idx]
    tc_str = subclust_info['test_cond']
    cl_str = str(int(subclust_info['clust_num']))
    sc_str = str(int(subclust_info['subclust']))
    # 06a. Generate a DataFrame for the annotations for each cassette's sub-blocks
    cassette_ann_df = pd.DataFrame(
            columns = [
                'sub_block_idx',
                'sb_DEG',
                'sb_response_ann',
                'sb_pos_reg_ann',
                'sb_neg_reg_ann',
                'sb_other_ann'
                ]
            )
    ca_idx = 0
    for sb_idx, sb_info in cassette_info.items():
        for sb_deg, go_info in sb_info.items():
            # 06a. Attempt to split the GO term annotations
            # into response-related information, directional
            # regulation information, and general information
            resp = []
            pos_reg = []
            neg_reg = []
            gen = []
            for go_curr in go_info:
                if pos_reg_info_tag in go_curr:
                    pos_reg.append(go_curr)
                elif neg_reg_info_tag in go_curr:
                    neg_reg.append(go_curr)
                elif response_info_tag in go_curr:
                    resp.append(go_curr)
                else:
                    gen.append(go_curr)
            pos_reg_str = ' AND '.join(
                    [
                        _.split(pos_reg_info_tag)[-1]
                        for _ in pos_reg
                        ]
                    )
            neg_reg_str = ' AND '.join(
                    [
                        _.split(neg_reg_info_tag)[-1]
                        for _ in neg_reg
                        ]
                    )
            resp_str = ' AND '.join(
                    resp
                    )
            gen_str = ' AND '.join(
                    gen
                    )
            cassette_ann_df.at[ca_idx,'sub_block_idx'] = sb_idx
            cassette_ann_df.at[ca_idx,'sb_DEG'] = sb_deg 
            cassette_ann_df.at[ca_idx,'sb_response_ann'] = resp_str
            cassette_ann_df.at[ca_idx,'sb_pos_reg_ann'] = pos_reg_str
            cassette_ann_df.at[ca_idx,'sb_neg_reg_ann'] = neg_reg_str
            cassette_ann_df.at[ca_idx,'sb_other_ann'] = gen_str
            # 06b. Figure out whether the current DEG is shared by
            # the complement test condition for the current test
            # condition, either in the complement's full DEG list
            # or in its own highly-coordinated DEG list.
            complement_tc = None
            sb_deg_no_dir = sb_deg.replace('_up_','_').replace('_down_','_')
            if tc_str in deg_overlap_check_tc_map.keys():
                complement_tc = deg_overlap_check_tc_map[tc_str]
                if complement_tc in all_degs_by_tc_dict.keys():
                    all_degs_c_tc = all_degs_by_tc_dict[complement_tc]
                    hcmct_degs_c_tc = hc_mct_degs_by_tc_dict[complement_tc]
                    cassette_ann_df.at[ca_idx,'shared_complement_full_DEGs'] = (
                            1 if sb_deg_no_dir in all_degs_c_tc
                            else 0
                            )
                    cassette_ann_df.at[ca_idx,'shared_complement_hc_degs'] = (
                            1 if sb_deg_no_dir in hcmct_degs_c_tc
                            else 0
                            )
                    ca_idx += 1
    # 06b. Save the DataFrame for the current cassette to file
    # with the appropriate DEG cluster/sub-cluster index information
    cassette_ann_fn = out_dir + f'/block_ann__{tc_str}__clust_{cl_str}__sc_{sc_str}.csv'
    cassette_ann_df.to_csv(
            cassette_ann_fn,
            index=True
            )

sys.exit()







