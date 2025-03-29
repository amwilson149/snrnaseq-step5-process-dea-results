import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys
import argparse
import yaml
import json
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering as AC

# This script takes a series of scipy dendrogram
# results dictionaries, one for each specified DEA,
# and performs subclustering on the dendrogram clusters.
# It does so by identifying points along the ordered
# cluster DEG sequence where the DEG pair distance
# metric exceeds some number of standard deviations
# from the mean (the mean of all ordered DEG pair
# distances in the cluster).

# Script setup

# 00. Create argparse object for reading input arguments
parser = argparse.ArgumentParser(description='Visualization of differentially expressed genes' + \
        ' for selected expression data partitions.')
parser.add_argument(f'--config-yaml-path',type=str,required=True)
args = parser.parse_args()

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
# 01a.iv. Cell type abbreviation map
ct_abbr_map = cfg.get('ct_to_abbreviation_dict')
# 01a.v. Output directory
ppt_out_dir = qc_root_dir + f'/' + cfg.get('deg_clust_data_dir')
if not os.path.isdir(ppt_out_dir):
    os.system(f'mkdir -p {ppt_out_dir}')
# 01a.vi. GO term dictionary directory
gene_set_dir = cfg.get('enr_dir')
# 01a.vii. GO term dictionary file
gene_set_fn_list = cfg.get('gseapy_enrichr_libs')

# 01b. DEA parameters
# 01b.i. DEAs for which to produce DEG subclusters
test_comparisons = cfg.get('dea_cond_comparison_list')
# 01b.ii. Significance level for DEG Pearson correlations
alpha = cfg.get('deg_pearson_corr_sig_alpha')
# 01b.iii. Flag indicating whether to process control
# group instead of treatment group data
to_do_controls_only = cfg.get('analyze_control_group_flag')
# 01b.iv. Tag for cell type mix of interest
ct_mix_tag = cfg.get('cell_type_mix_tag')
# 01b.v. RNG random seed
rand_seed = cfg.get('rand_seed_deg_corr')
np.random.seed(seed=rand_seed)
# 01b.vi. File name tag for DEG cluster label files
deg_clust_lbl_fn_tag = cfg.get('deg_clust_label_fn_tag')
# 01b.vii. Dispersion threshold to use for identifying
# putative subcluster junctions
disp_threshold = cfg.get('ppt_deg_clust_disp_threshold')
# 01b.viii. Output file name tag (with clearer DEA description)
deg_comp_out_fn_tag = cfg.get('deg_comp_tc_fn_tag')

# 01c. DEG subcluster output directory information
subclust_out_dir = ppt_out_dir + f'/' + cfg.get('deg_sc_data_dir') + f'/{ct_mix_tag}'
if to_do_controls_only:
    subclust_out_dir += f'/control_group'
else:
    subclust_out_dir += f'/treatment_group'
if not os.path.isdir(subclust_out_dir):
    os.system(f'mkdir -p {subclust_out_dir}')

# 01d. Flag specifying whether to print the DEG
# correlation values as text on DEG subcluster
# heatmap plots
add_txt = cfg.get('add_htmap_txt')

# 02. Read in dendrogram dictionaries, 
# DEG cluster labels, and the DataFrames
# listing the significant pairwise DEG
# Pearson correlations for each specified DEA
tc_dend_dict = {}
tc_cell_type_dict = {}
tc_clust_lbl_dict ={}
tc_all_sig_corr_df = pd.DataFrame()
for tc_curr in test_comparisons:
    tc_str = f'{tc_curr[0]}: {tc_curr[1]} vs. {tc_curr[2]}'
    print(f'Getting DEG correlation info for\n\t{tc_str}')
    tc_out_str = '_'.join(tc_curr)
    in_dir_curr = f'{ppt_out_dir}/{tc_out_str}/{ct_mix_tag}'
    if to_do_controls_only:
        in_dir_curr += f'/control_group'
    else:
        in_dir_curr += f'/treatment_group'
    in_dir_clust_curr = in_dir_curr + '/' + f'similarity_plots'

    # 02a. Get dendrogram file name
    dend_fn = [
            _ for _ in
            os.listdir(in_dir_clust_curr)
            if
            (
                ('dendrogram' in _)
                &
                ('.json' in _)
                )
            ]
    if len(dend_fn) > 0:
        dend_fn = dend_fn[0]
        dend_full_fn = f'{in_dir_clust_curr}/{dend_fn}'
        with open(dend_full_fn,'r') as infile:
            tc_dend_dict[tc_out_str] = json.load(
                    infile
                    )

    # 02b. Get DEG cluster label file
    deg_clust_lbl_fn = [
            _ for _ in
            os.listdir(in_dir_clust_curr)
            if 
            (
                (deg_clust_lbl_fn_tag in _)
                &
                ('.csv' in _)
                )
            ]
    if len(deg_clust_lbl_fn) > 0:
        deg_clust_lbl_fn = deg_clust_lbl_fn[0]
        deg_clust_lbl_full_fn = f'{in_dir_clust_curr}/{deg_clust_lbl_fn}'
        dcl_df = pd.read_csv(
                deg_clust_lbl_full_fn,
                index_col=0
                )
        cl_id_to_num = [
                (cl_color,cl_num,)
                for cl_color, cl_num in
                zip(
                    dcl_df['cluster_id'].values.tolist(),
                    dcl_df['cluster_id_num'].values.tolist()
                    )
                ]
        cl_id_to_num = list(set(cl_id_to_num))
        tc_clust_lbl_dict[tc_out_str] = {
                _[0]:_[1]
                for _ in cl_id_to_num
                }
    # 02c. Get pairwise correlation DataFrame
    pw_corr_fn = [
            _ for _ in
            os.listdir(in_dir_curr)
            if 
            (
                ('deg_corr_w_sig' in _)
                &
                ('.csv' in _)
                )
            ]
    if len(pw_corr_fn) > 0:
        pw_corr_fn = pw_corr_fn[0]
        pw_corr_full_fn = f'{in_dir_curr}/{pw_corr_fn}'
        pw_corr_df = pd.read_csv(
                pw_corr_full_fn,
                index_col=0
                )
        pw_corr_df['test_cond'] = [tc_out_str]*len(pw_corr_df)
        if len(tc_all_sig_corr_df) == 0:
            tc_all_sig_corr_df = pw_corr_df.copy()
        else:
            tc_all_sig_corr_df = pd.concat(
                    [
                        tc_all_sig_corr_df,
                        pw_corr_df.copy()
                        ],
                    ignore_index=True
                    )
    # 02c.i. Find all cell types in the current DEA
    tc_curr_all_degs = list(
            set(
                pw_corr_df['deg_1'].values.tolist()
                ).union(
                    set(
                        pw_corr_df['deg_2'].values.tolist()
                        )
                    )
                )
    tc_curr_cell_types = list(
            set(
                [
                    _.split('_')[0]
                    for _ in
                    tc_curr_all_degs
                    ]
                )
            )
    tc_cell_type_dict[tc_out_str] = sorted(tc_curr_cell_types)

# 02d. Add a column in the Pearson correlation DataFrame
# containing the distance measure used to 
# generate the dendrogram to facilitate interpretation
# of similarities among the DEGs in the cluster
tc_all_sig_corr_df['abs_sig_pc_dist'] = [
        1.0/(1.0+np.abs(_))
        for _ in
        tc_all_sig_corr_df['pearsons_corr_w_beta_test'].values.tolist()
        ]

# 02e. Get cell types for which to read in GSEApy data
cell_types_abbrev = list(
        set(
            [
                item
                for _ in tc_cell_type_dict.values()
                for item in _
                ]
            )
        )
cell_type_back_map = {}
for cta in cell_types_abbrev:
    keycurr =[
            k for k,v in ct_abbr_map.items()
            if cta==v
            ][0]
    cell_type_back_map[cta] = keycurr

# 03. Identify putative subclusters for each
# cluster, for each specified DEA
subclust_info_df = pd.DataFrame()
sss_idx = 0
for tc_curr in tc_dend_dict.keys():
    # 03a. Get the dendrogram, cluster label map,
    # correlation data, and GSEApy results
    # for the current DEA
    dend = tc_dend_dict[tc_curr]
    clust_lbl = tc_clust_lbl_dict[tc_curr]
    corr = tc_all_sig_corr_df.loc[
            tc_all_sig_corr_df['test_cond'] == tc_curr
            ].copy()
    cell_types_curr = [
            cell_type_back_map[_]
            for _ in
            tc_cell_type_dict[tc_curr]
            ]
    # 03a.i. Set up an output directory for DEG subcluster
    # Pearson correlation heatmaps
    plot_out_dir_curr = f'{subclust_out_dir}/{tc_curr}__subcluster_plots'
    if not os.path.isdir(plot_out_dir_curr):
        os.system(f'mkdir -p {plot_out_dir_curr}')

    # 03b. Set up a 'subcluster_label' field
    # for the current dendrogram
    dend_subclust_labels = np.zeros(
            (len(dend['ivl']),1)
            )

    # 03c. Iterate through DEG clusters and perform splitting
    for clust_color,clust_num in clust_lbl.items():
        print(clust_color)
        print(clust_num)
        # 03b.i. Get current dendrogram cluster
        dend_clust_idx = [
                i for i,_ in
                enumerate(dend['leaves_color_list'])
                if _==clust_color
                ]
        # 03c.ii. Get cluster DEGs
        dend_clust_degs = [
                dend['ivl'][i]
                for i in dend_clust_idx
                ]
        # 03c.iii. Generate a 'direction-agnostic'
        # version of cluster DEG names for downstream
        # analyses
        dend_clust_degs_no_dir = [
                _.replace('_down_','_').replace('_up_','_')
                for _ in
                dend_clust_degs
                ]
        # 03c.iv. Get the distances and similarities
        # for ordered cluster DEGs
        similarities = []
        distances = []
        for pair_idx in range(
                len(
                    dend_clust_degs
                    )-1
                ):
            deg_1_curr = dend_clust_degs[pair_idx]
            deg_2_curr = dend_clust_degs[pair_idx+1]
            corr_row = corr.loc[
                    corr[['deg_1','deg_2']].isin(
                        [deg_1_curr,deg_2_curr]
                        ).all(axis=1)
                    ]
            similarities.append(
                    np.abs(
                        corr_row['pearsons_corr_w_beta_test'].values.tolist()[0]
                        )
                    )
            distances.append(
                    corr_row['abs_sig_pc_dist'].values.tolist()[0]
                    )
        # 03c.v. Compute the dispersion (n standard deviations
        # from mean of ordered pairwise distances)
        if len(distances) > 0:
            mean_curr = np.mean(distances)
            std_curr = np.std(distances)
            # 03c.v.A. Handle the case of 0 cluster standard deviation
            # (where all ordered DEG pair distances are equal)
            # so that all DEG pair dispersions are 0
            if std_curr == 0.0:
                print('std. dev. of current clust. is 0; setting to 1 now')
                std_curr = 1.0
            # 03c.v.B. Handle case of nan mean or standard
            # devation: print the values to see what the issue is
            if mean_curr != mean_curr:
                print(f'cluster contains nan values. printing distances:')
                dist_str = ', '.join([str(_) for _ in distances])
                print(f'\t{dist_str}')
                print(f'printing DEGs:')
                deg_str = ', '.join(dend_clust_degs)
                print(f'\t{deg_str}')
            print(f'mean: {mean_curr}, std.: {std_curr}')
            disp = [
                    (_-mean_curr)*1.0/std_curr
                    for _ in distances
                    ]
                
            # 03c.vi. Find the locations where DEG pair distances
            # exceed the specified dispersion threshold
            over_thresh_disp_idxs = [
                    i for i,_ in enumerate(disp)
                    if _ > disp_threshold
                    ]

            # 03c.vii. Split the cluster into subclusters
            # at these locations
            subclusts = np.zeros(
                    (len(dend_clust_degs),1)
                    )
            sc_label_curr = 0
            start_idxs = [0] + [_+1 for _ in over_thresh_disp_idxs]
            end_idxs = [_+1 for _ in over_thresh_disp_idxs] + [len(disp)+1]
            for s_idx,e_idx in zip(start_idxs,end_idxs):
                # 03c.vii.A. Add labels to subcluster label array
                subclusts[s_idx:e_idx,0] = sc_label_curr
                degs_subclust_curr = dend_clust_degs[s_idx:e_idx]
                # 03c.vii.B. Record DEG subcluster average similarity and
                # other metadata
                sims_curr = similarities[s_idx:e_idx-1]
                subclust_info_df.at[sss_idx,'test_cond'] = tc_curr
                subclust_info_df.at[sss_idx,'clust_color'] = clust_color
                subclust_info_df.at[sss_idx,'clust_num'] = clust_num
                subclust_info_df.at[sss_idx,'subclust'] = sc_label_curr
                subclust_info_df.at[sss_idx,'n_degs_subclust'] = len(degs_subclust_curr)
                subclust_info_df.at[sss_idx,'degs_subclust'] = ','.join(degs_subclust_curr)
                subclust_info_df.at[sss_idx,'mean_similarity'] = np.mean(sims_curr)
                # 03c.vii.C. Generate a correlation plot for the current subcluster
                corr_arr = np.identity(
                        len(degs_subclust_curr),
                        dtype=np.float16
                        )
                for dsc_idx1 in range(0,len(degs_subclust_curr)-1):
                    dsc1 = degs_subclust_curr[dsc_idx1]
                    for dsc_idx2 in range(dsc_idx1+1,len(degs_subclust_curr)):
                        dsc2 = degs_subclust_curr[dsc_idx2]
                        corr_row_curr = corr.loc[
                                corr[['deg_1','deg_2']].isin(
                                    [dsc1,dsc2]
                                    ).all(axis=1)
                                ]
                        corr_curr = corr_row_curr['pearsons_corr_w_beta_test'].values.tolist()[0]
                        corr_arr[dsc_idx1,dsc_idx2] = corr_curr
                        corr_arr[dsc_idx2,dsc_idx1] = corr_curr
                fig,ax = plt.subplots(
                        figsize=(8,7.2)
                        )
                im = plt.imshow(
                        corr_arr,
                        cmap='inferno',
                        vmin=-1.0,
                        vmax=1.0
                        )
                for spine in im.axes.spines.values():
                    spine.set_edgecolor(clust_color)
                im.axes.tick_params(color=clust_color)
                degs_subclust_curr_plot = [
                        _.replace('_',' ').replace('down',f'\u2193').replace('up',f'\u2191')
                        for _ in
                        degs_subclust_curr
                        ]
                fontsize_arg = ('xx-small' if len(degs_subclust_curr) > 15 else 'small')
                ax.set_xticks(
                        np.arange(len(degs_subclust_curr)),
                        degs_subclust_curr_plot,
                        ha='right',
                        rotation=45,
                        rotation_mode='anchor',
                        fontsize=4
                        )
                ax.set_yticks(
                        np.arange(len(degs_subclust_curr)),
                        degs_subclust_curr_plot,
                        ha='right',
                        va='center',
                        fontsize=4
                        )
                # 03c.vii.D. Add text annotations with correlation values if specified
                if add_txt:
                    fontsize_arg = ('xx-small' if len(degs_subclust_curr) > 15 else 'small')
                    for idx1_txt in range(corr_arr.shape[0]):
                        for idx2_txt in range(corr_arr.shape[1]):
                            corr_val = corr_arr[idx1_txt,idx2_txt]
                            corr_val_color = ('w' if corr_val < 0.25 else 'k')
                            corr_val_color = ('g' if corr_val==1.0 else corr_val_color)
                            corr_str = (
                                    f'{corr_val:0.3f}'
                                    if len(degs_subclust_curr)<50
                                    else
                                    f'{corr_val:0.2f}'
                                    )
                            fontsize_arg_txt = (
                                    4
                                    if len(degs_subclust_curr)<50
                                    else
                                    3
                                    )
                            rotation_arg_txt = (
                                    0
                                    if len(degs_subclust_curr)<50
                                    else
                                    -45
                                    )
                            text_curr = ax.text(
                                    idx2_txt,
                                    idx1_txt,
                                    corr_str,
                                    ha='center',
                                    va='center',
                                    color=corr_val_color,
                                    fontsize=fontsize_arg_txt
                                    )
                # 03c.vii.E. Add a colorbar
                cbar = ax.figure.colorbar(
                        im,
                        ax=ax,
                        fraction=0.02,
                        shrink=0.6,
                        aspect=40,
                        pad=0.015,
                        )
                cbar.ax.tick_params(labelsize=fontsize_arg)
                cbar_label = 'Cross-Patient Pearson Corr.'
                cbar.ax.set_ylabel(
                        cbar_label,
                        rotation=-90,
                        va='bottom',
                        fontsize=4
                        )
                fig.tight_layout()
                plt.title(
                        f'Significant Pearson Correlations, Cluster {int(clust_num)}, Subcluster {int(sc_label_curr)}',
                        fontsize=4
                        )
                add_txt_lbl = ('_with_txt' if add_txt else '')
                plt_fn = f'{plot_out_dir_curr}/{tc_curr}__deg_corrs_clust_{clust_num}_subclust_{sc_label_curr}{add_txt_lbl}.png'
                plt.savefig(
                        plt_fn,
                        bbox_inches='tight',
                        dpi=500
                        )
                plt.close('all')
                # 03c.vii.F. Increment DEG subcluster information
                # Dataframe row index and DEG subcluster label
                sss_idx += 1
                sc_label_curr += 1


            # 03c.viii. Convert sublcluster labels to a list
            # and add them to the current dendrogram's subcluster field
            scl = [item for _ in subclusts.tolist() for item in _]
            for sc_idx in range(len(scl)):
                dend_subclust_labels[
                        dend_clust_idx[sc_idx],
                        0
                        ] = scl[sc_idx]
        else:
            idx_str = ', '.join(
                    [
                        str(_) for i,_ in
                        enumerate(dend_clust_idx)
                        ]
                    )
            print(f'only 1 DEG is in this cluster; assigning subcluster label 0 ' + \
                    f'at dendrogram index {idx_str}')
            scl = [0]
            degs_subclust_curr = dend_clust_degs
            subclust_info_df.at[sss_idx,'test_cond'] = tc_curr
            subclust_info_df.at[sss_idx,'clust_color'] = clust_color
            subclust_info_df.at[sss_idx,'clust_num'] = clust_num
            subclust_info_df.at[sss_idx,'subclust'] = scl[0]
            subclust_info_df.at[sss_idx,'n_degs_subclust'] = len(degs_subclust_curr)
            subclust_info_df.at[sss_idx,'degs_subclust'] = ','.join(degs_subclust_curr)
            subclust_info_df.at[sss_idx,'mean_similarity'] = np.nan
            sss_idx += 1

        # 03d. Add subcluster labels to the current dendrogram array
        for sc_idx in range(len(scl)):
            dend_subclust_labels[
                    dend_clust_idx[sc_idx],
                    0
                    ] = scl[sc_idx]

    # 04. Set dendrogram subcluster label field
    dend_scl = [item for _ in dend_subclust_labels.tolist() for item in _]
    dend['ivl_subclust'] = dend_scl

# 05. Save the DataFrame with subcluster information to file
all_subclust_df_out_fn = f'{subclust_out_dir}/{deg_comp_out_fn_tag}__test_cond_deg_subclust_w_scores.csv'
subclust_info_df.to_csv(
        all_subclust_df_out_fn,
        index=True
        )

sys.exit()


