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

# This script performs hierarchical clustering
# of DEGs from their significant pairwise Pearson
# correlations, with similarity = abs(Pearson correlation)
# if it is significant and 0 otherwise, and
# with the distance = 1.0/(1.0+similarity).

# Script setup

# 00. Create argparse object for reading input arguments
parser = argparse.ArgumentParser(description='Visualization of differentially expressed genes' + \
        ' for selected expression data partitions.')
parser.add_argument(f'--config-yaml-path',type=str,required=True)
args = parser.parse_args()

# 00a. Get color dictionary
color_dict = mcolors.CSS4_COLORS
# 00a.i. Pull darker colors only
# 00a.i.A. Create a list of light
# color tags by manually inspecting
# the keys, as displayed in the
# reference documentation at
# https://matplotlib.org/stable/
# gallery/color/named_colors.html
light_color_tags = [
        'light',
        'white',
        'snow',
        'mistyrose',
        'seashell',
        'peachpuff',
        'linen',
        'bisque',
        'whip',
        'chiffon',
        'lace',
        'ivory',
        'beige',
        'honeydew',
        'cream',
        'azure',
        'aliceblue',
        'thistle',
        'lavenderblush'
        ]
# 00a.i.B. Create a list of hard-
# to-distinguish color tags to
# remove
indist_color_tags = [
        'gray',
        'grey',
        'silver',
        'gainsboro',
        'salmon',
        'midnightblue',
        'darkblue',
        'medium',
        'blue',
        'orchid',
        'magenta',
        'cyan',
        'aqua',
        'yellow',
        'lawngreen',
        'chartreuse',
        'lime',
        'spring'
        ]
# 00a.i.C. Build a full list
# of bad color tags
bad_color_tags = light_color_tags + \
        indist_color_tags
# 00a.ii. Pick out the color
# keys that don't have any
# of these tags in them
darker_color_keys = [
        _ for _ in color_dict.keys()
        if not any(
            [
                tag in _
                for tag in
                bad_color_tags
                ]
            )
        ]

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
# 01a.vi. Folder for cell-type-mix-specific results
cell_type_mix_tag = cfg.get('cell_type_mix_tag')

# 01b. DEA parameters
# 01b.i. DEAs for which to assess DEG correlations
test_comparisons = cfg.get('dea_cond_comparison_list')
# 01b.ii. Significance level for DEG Pearson correlations
alpha = cfg.get('deg_pearson_corr_sig_alpha')
# 01b.iii. Flag indicating whether to process control
# instead of treatment group data
to_do_controls_only = cfg.get('analyze_control_group_flag')
# 01b.iv. RNG random seed
rand_seed = cfg.get('rand_seed_deg_corr')
np.random.seed(seed=rand_seed)

# 01c. Colormap post-processing
# 01c.i Randomize darker color keys to avoid
# consecutive similar colors
randomized_key_idxs = np.random.permutation(
        np.arange(len(darker_color_keys))
        )
darker_color_keys = [
        darker_color_keys[_]
        for _ in randomized_key_idxs
        ]
# 01c.ii. Get the resulting dictionary
# of darker colors for plotting
darker_color_dict = {
        i:k
        for i,k in
        enumerate(darker_color_keys)
        }

# 01d. Define a helper function for plotting dendrograms,
# follow scikit-learn's documentation:
# https://scikit-learn.org/stable/auto_examples/cluster/
# plot_agglomerative_dendrogram.html#sphx-glr-auto-
# examples-cluster-plot-agglomerative-dendrogram-py
def plot_dendrogram(model, **kwargs):
    global darker_color_dict
    # 01d.i. Create linkage matrix and then plot the dendrogram
    # 01d.i.A. Create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    
    linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
            ).astype(float)
    # 01d.i.B. Plot the corresponding dendrogram
    hierarchy.set_link_color_palette(
            [
                v for v
                in darker_color_dict.values()
                ]
            )
    dn_data = hierarchy.dendrogram(
            linkage_matrix,
            **kwargs
            )
    hierarchy.set_link_color_palette(None)
    return dn_data

# 02. Iterate over the test condition comparisons,
# pull the DEG correlations, and perform clustering
# based on their similarities
for tc_curr in test_comparisons:
    tc_str = f'{tc_curr[0]}: {tc_curr[1]} vs. {tc_curr[2]}'
    print(f'Assessing DEG correlations for\n\t{tc_str}')
    tc_out_str = '_'.join(tc_curr)
    in_dir_curr = f'{ppt_out_dir}/{tc_out_str}/{cell_type_mix_tag}'
    if to_do_controls_only:
        in_dir_curr += f'/control_group'
    else:
        in_dir_curr += f'/treatment_group'
    out_dir_curr = in_dir_curr + '/' + f'similarity_plots'
    if not os.path.isdir(out_dir_curr):
        os.system(f'mkdir -p {out_dir_curr}')
    # 02a. Get the DEG .csv file for the current comparison
    deg_corr_fn = [
            _ for _ in
            os.listdir(in_dir_curr)
            if '.csv' in _
            ][0]
    deg_corr_full_fn = in_dir_curr + '/' + deg_corr_fn
    deg_corr_df = pd.read_csv(
            deg_corr_full_fn,
            index_col=0
            )

    # 02b. Retain only statistically significant DEG correlations
    deg_corr_sig_df = deg_corr_df.loc[
            deg_corr_df[f'sig_alpha_beta_assumption_{alpha}']>0.0
            ].copy()
    pc_sig = len(deg_corr_sig_df)*100.0/len(deg_corr_df)
    print(f'\tExtracted {len(deg_corr_sig_df)} significantly correlated ' + \
            f'DEG pairs\n\t({pc_sig:0.02f}% of all pairs)')

    # 02c. Build a similarity matrix using the significant correlations,
    # where the similarity is abs(Pearson correlation) if the correlation
    # is significant and 0 otherwise
    all_degs = list(set(
        deg_corr_df['deg_1'].values.tolist() + \
                deg_corr_df['deg_2'].values.tolist()
                ))
    print(f'\tBuilding similarity matrix for {len(all_degs)} DEGs, ' + \
            f'using only significant pairwise Pearson correlations.')
    sig_corr_abs_mtx = np.identity(
            len(all_degs),
            dtype=np.float16
            )
    sig_corr_sign_mtx = np.identity(
            len(all_degs),
            dtype=np.int8
            )
    sig_corr_mtx = np.identity(
            len(all_degs),
            dtype=np.float16
            )
    # 02d.i. Generate a 1D version of these similarities
    # to use as input for the hierarchical clustering
    # linkage function
    # 02d.i.A Generate an (n-choose-2) x 1 array, where
    # n is the total number of DEGs
    # Note: the ordering of this 1D matrix should
    # be the same as that returned by the
    # scipy.spatial.distance.pdist function (see
    # https://docs.scipy.org/doc/scipy/reference/
    # generated/scipy.spatial.distance.pdist.html),
    # i.e. that for indices i < j < m, where m
    # is the total *number* of observations (each
    # with an n-dimensional measurement; for us
    # each DEG is an observation and it consists
    # of counts in n-dimensional "patient" space)
    # i.e. that dist(u=X[i],v=X[j]) (or for us,
    # just dist(i,j) = 1.0/(1.0+sim(i,j))
    # is stored in entry
    # m * i + j - ((i+2) * (i+1)) // 2
    # (Note that the // operator performs
    # floor division).
    sig_corr_abs_1d = np.zeros(
            (
                len(all_degs)*(len(all_degs)-1)//2,
                1
                )
            )
    sig_corr_sign_1d = np.zeros(
            (
                len(all_degs)*(len(all_degs)-1)//2,
                1
                )
            )
    m_ = len(all_degs)
    for idx in deg_corr_sig_df.index.values.tolist():
        row = deg_corr_sig_df.loc[idx]
        d1 = row['deg_1']
        d2 = row['deg_2']
        sim = row['pearsons_corr_w_beta_test']
        d1_idx = [
                i for i,_ in enumerate(all_degs)
                if _==d1
                ][0]
        d2_idx = [
                i for i,_ in enumerate(all_degs)
                if _==d2
                ][0]
        sig_corr_abs_mtx[d1_idx,d2_idx] = np.abs(sim)
        sig_corr_abs_mtx[d2_idx,d1_idx] = np.abs(sim)
        sig_corr_sign_mtx[d1_idx,d2_idx] = (-1 if sim<0 else 1)
        sig_corr_sign_mtx[d2_idx,d1_idx] = (-1 if sim<0 else 1)
        sig_corr_mtx[d1_idx,d2_idx] = sim
        sig_corr_mtx[d2_idx,d1_idx] = sim
        # 02d.ii. Compute the location in the 1D similarity array
        # for this similarity
        i_ = np.min([d1_idx,d2_idx])
        j_ = np.max([d1_idx,d2_idx])
        loc_ = (m_*i_) + j_ - ((i_+2)*(i_+1))//2
        sig_corr_abs_1d[loc_,] = np.abs(sim)
        sig_corr_sign_1d[loc_,] = (-1 if sim<0 else 1)
    # 02d.iii. Make sure all similarity matrices have
    # the correct dtype
    sig_corr_abs_mtx = sig_corr_abs_mtx.astype(np.float16)
    sig_corr_sign_mtx = sig_corr_sign_mtx.astype(np.int8)
    sig_corr_mtx = sig_corr_mtx.astype(np.float16)
    sig_corr_abs_1d = sig_corr_abs_1d.astype(np.float16)
    sig_corr_sign_1d = sig_corr_sign_1d.astype(np.int8)

    # 02e. Generate a distance matrix for all 
    # significantly correlated DEGs, where the
    # magnitude of the correlation
    # is transformed to a distance via the formula
    # dist = 1.0/(1.0+np.abs(corr)).
    dist_mtx = np.divide(
            np.ones(
                (
                    len(all_degs),
                    len(all_degs)
                    )
                ).astype(np.float16),
            np.add(
                np.ones(
                    (
                        len(all_degs),
                        len(all_degs)
                        )
                    ).astype(np.float16),
                sig_corr_abs_mtx
                )
            ).astype(np.float16)

    # 03. Perform agglomerative hierarchical clustering
    # using scikit-learn (which handles larger datasets
    # better than scipy), with average linkage distance.
    agglom_clust = AC(
            distance_threshold=0,
            n_clusters=None,
            metric='precomputed',
            linkage='average'
            ).fit(
            dist_mtx
            )

    # 03b. Plot dendrogram with clustering results
    fig,(ax1,ax2) = plt.subplots(2,1,gridspec_kw={'height_ratios':[4,1]})
    dend_results = plot_dendrogram(
            agglom_clust,
            labels=all_degs,
            above_threshold_color='gray',
            distance_sort='ascending',
            ax=ax1
            )
    ax1.set_xticks([])
    ax1.set_yticks([])
    # 03b.i. Save the dendrogram results to file
    dend_full_out_fn =  f'{out_dir_curr}/dendrogram_results_dict.json'
    with open(dend_full_out_fn,'w') as outfile:
        json.dump(
                dend_results,
                outfile
                )

    # 03b.ii. Pull the ordered list of leaf colors
    # for clustered data for downstream analysis,
    # as well as the unique leaf colors (which
    # corresponds to the number of automatically
    # generated clusters)
    leaf_ordered_labels = dend_results['ivl']
    leaf_colors = dend_results['leaves_color_list']

    # 03b.iii. Build a DataFrame with the unique members
    # of each ordered color (cluster) in the dendrogram
    leaf_data_df = pd.DataFrame(
            data = {
                'DEG': leaf_ordered_labels,
                'cluster_id': leaf_colors
                }
            )

    # 03c. Use the leaf colors to generate a dendrogram
    # legend and to break up the ordered DEGs into clusters
    # for the dendrogram and to generate per-cluster
    # 03c.i. Get the ordered list of colors (different from above)
    link_color_list = dend_results['color_list']
    lc_unique = []
    for link_color in link_color_list:
        if link_color not in lc_unique:
            lc_unique.append(link_color)
    # 03c.i.A. Add the numeric cluster ID to the leaf
    # DataFrame using the unique leaf color list
    for lf_idx in leaf_data_df.index.values.tolist():
        row = leaf_data_df.loc[lf_idx]
        clust_color_curr = row['cluster_id']
        leaf_data_df.at[lf_idx,'cluster_id_num'] = [
                i for i,_ in
                enumerate(lc_unique)
                if _==clust_color_curr
                ][0]
    # 03c.i.B. Save DEG cluster information
    leaf_data_out_fn = f'{out_dir_curr}/deg_clust_labels.csv'
    leaf_data_df.to_csv(
            leaf_data_out_fn,
            index=True
            )

    # 03c.ii. Generate a colorbar
    color_mtx = np.arange(len(lc_unique))[np.newaxis,:]
    cmap_cmtx = mcolors.ListedColormap(lc_unique)
    ax2.imshow(
            color_mtx,
            cmap_cmtx
            )
    ax2.set_xticks(
            np.arange(len(lc_unique)),
            [f'Clust\n{i}' for i in range(len(lc_unique))],
            ha='right',
            rotation=45,
            rotation_mode='anchor',
            fontsize=4
            )
    ax2.set_yticks([])
    plt.subplots_adjust(
            wspace=None,
            hspace=0.01
            )
    fig.suptitle(
            f'DEG clustering, Pearson-correlation-based distances' + \
                    f'(most similar -> least similar)',
                    fontsize=10
                    )
    dend_fn = f'{out_dir_curr}/pearson_corr_abs_dist_dendrogram.png'
    plt.savefig(
            dend_fn,
            dpi=300,
            bbox_inches='tight'
            )
    plt.close('all')

    # 03c. Plot the correlation heatmap, and write each cluster's
    # list of DEGs to an Excel file
    writer_dict = {}
    print(f'\tGetting the DEG members of each correlation cluster...')
    for lc,deg_df in leaf_data_df.groupby('cluster_id'):
        clust_idx = [
                i
                for i,_ in enumerate(lc_unique)
                if _==lc
                ][0]
        print(f'\t\tCluster {clust_idx} ({lc})')
        degs_clust = deg_df['DEG'].values.tolist()
        dc_str = '\n\t\t\t'.join(degs_clust)
        print(f'\t\t\tDEG members:\n\t\t\t{dc_str}')

        # 03c.i. Save the current DEG cluster information
        # to a subdirectory depending on what cell type(s)
        # its DEGs correspond to
        cell_types_clust = sorted(
                list(set([
                    _.split('_')[0]
                    for _ in degs_clust
                    ]))
                )
        cell_types_clust_str = '_'.join(cell_types_clust)
        out_dir_clust_type_curr = out_dir_curr + '/' + cell_types_clust_str
        if not os.path.isdir(out_dir_clust_type_curr):
            os.system(f'mkdir -p {out_dir_clust_type_curr}')
        # 03c.i.A. Initialize Excel file writer
        if cell_types_clust_str not in writer_dict.keys():
            clust_type_deg_full_fn = f'{out_dir_clust_type_curr}/deglists_clusters_w_cell_type_{cell_types_clust_str}.xlsx'
            writer_dict[cell_types_clust_str] = pd.ExcelWriter(clust_type_deg_full_fn)

        # 03c.ii. Get current DEG cluster heatmap
        sim_clust_curr = sig_corr_mtx.copy()
        idxs_degs_clust = [
                i for i,_ in enumerate(all_degs)
                if _ in degs_clust
                ]
        sim_clust_curr = sim_clust_curr[idxs_degs_clust,:].copy()
        sim_clust_curr = sim_clust_curr[:,idxs_degs_clust].copy()
        fig,ax = plt.subplots()
        im = plt.imshow(
                sim_clust_curr,
                cmap='inferno',
                vmin=-1.0,
                vmax=1.0
                )
        # 03c.ii.A. Set heatmap outline color to match what is
        # in the dendrogram
        for spine in im.axes.spines.values():
            spine.set_edgecolor(lc)
        # 03c.ii.B. Set axis tick label positions and colors
        ax.set_xticks(
                np.arange(len(idxs_degs_clust)),
                degs_clust,
                ha='right',
                rotation=45,
                rotation_mode='anchor',
                fontsize=4
                )
        ax.set_yticks(
                np.arange(len(idxs_degs_clust)),
                degs_clust,
                ha='right',
                va='center',
                fontsize=4
                )
        im.axes.tick_params(color=lc)
        cbar = ax.figure.colorbar(
                im,
                ax=ax
                )
        cbar_label = 'Cross-Patient Pearson Corr.'
        cbar.ax.set_ylabel(
                cbar_label,
                rotation=-90,
                va='bottom'
                )
        fig.tight_layout()
        plt.title(f'Significant Pearson Correlations,\nCluster {clust_idx} DEGs')
        plt_fn = f'{out_dir_clust_type_curr}/deg_corrs_clust_{clust_idx}.png'
        plt.savefig(
                plt_fn,
                dpi=300
                )
        plt.close()

        # 03d. Write current cluster DEG list into a new sheet
        # of the Excel document
        degs_only_df = deg_df.copy().drop(
                columns='cluster_id'
                )
        degs_only_df.to_excel(
                writer_dict[cell_types_clust_str],
                sheet_name=f'Cluster_{clust_idx}',
                startrow=0,
                startcol=0
                )

    # 03e. Save and close all ExcelWriter objects
    for ew in writer_dict.values():
        ew.close()



sys.exit()


