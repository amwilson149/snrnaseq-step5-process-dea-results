import numpy as np
import pandas as pd
import anndata as ad
import os
import sys
import argparse
import yaml
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from utils.plt_funcs import *

# This script takes the DEG lists for specified comparisons
# and generates visualizations of DEG intersections and
# differences.
# This script parses up- and down-regulated DEGs separately.

# Script setup

# 00. Create argparse object for reading input arguments
parser = argparse.ArgumentParser(description='Visualization of differentially expressed genes' + \
        ' for selected expression data partitions.')
parser.add_argument(f'--config-yaml-path',type=str,required=True)
parser.add_argument(f'--group-to-process',type=str,required=True)
args = parser.parse_args()

# 00b. Set text plot parameters
# 00b.i. Set font size for plots
plt.rcParams['font.size']=8

# 01. Get analysis parameters from configuration file
cfg = None
with open(args.config_yaml_path) as f:
    cfg = yaml.safe_load(f)
# 01a. Dataset information
# 01a.i. Data root directory
qc_root_dir = cfg.get('root_dir')
# 01a.ii. Differential expression analysis output directories
# 01a.ii.A. Parent directory with example preprocessed DEG lists
dea_data_dir = qc_root_dir + f'/' + cfg.get('dea_comparison_data_dir')
# 01a.ii.B. Subdirectories to DEA results files
dea_diffexpr_dir = dea_data_dir + f'/' + cfg.get('dea_deseq_formatted_data_dir')
dea_results_dir = dea_diffexpr_dir + f'/' + cfg.get('dea_results_dir')
# 01a.iii. DEG list comparison output directory
output_data_dir = qc_root_dir + f'/' + cfg.get('output_dea_data_dir')
output_diffexpr_dir = output_data_dir + f'/' + cfg.get('dea_deseq_formatted_data_dir')
output_results_dir = output_diffexpr_dir + f'/' + cfg.get('dea_results_dir')
dea_comp_out_dir = output_results_dir + f'/' + cfg.get('dea_set_comp_dir')
if not os.path.isdir(dea_comp_out_dir):
    os.system(f'mkdir -p {dea_comp_out_dir}')

# 01b. DEA parameters for visualization
# 01b.i. Test comparisons
test_comparisons = cfg.get('dea_cond_comparison_list')
# 01b.ii. Name tag for files with
# significantly upregulated genes
gene_df_upreg_root = cfg.get('dea_output_upreg_fn_root')
# 01b.iii. Name tag for files with
# significantly downregulated genes
gene_df_downreg_root = cfg.get('dea_output_downreg_fn_root')
# 01b.iv. The adjusted alpha used for DEA
gene_dea_alpha = cfg.get('dea_alpha')
# 01b.v. Name tag in columns with adjusted
# p-values for fold-removed log2fc values
dea_p_col_tag = cfg.get('dea_p_col_tag')
# 01b.vi. Optional mapping of test conditions
# into subsets for which individual venn
# diagrams and comparisons will be made
# NOTE: Visualizations with heatmaps will be
# built by taking comparison grous, then comparisons,
# from top to bottom, and by filling each row
# of heatmap from bottom to top; please ensure
# that the test condition architecture is
# ordered to produce the desired visualization
# given this organizing principle
tc_subset_mapping = cfg.get('tc_subset_mapping')
if tc_subset_mapping == 'None':
    tc_subset_mapping = {}
    tc_subset_mapping[''] = test_comparisons
# 01b.vii. Get the dictionary specifying the 
# test conditions for which to find and report
# overlaps, by group
group_ov_pair_dict = cfg.get('group_overlap_pair_dict')
# 01b.viii. Dictionary with HIV status text map
HIV_map = cfg.get('hiv_txt_dict')
# 01b.ix. Dictionary with SUD status text map
SUD_map = cfg.get('sud_txt_dict')

# 01c. Specify column with cross-fold-averaged
# log2 fold change values for DEGs
l2fc_col = 'cross_fold_l2fc_mean'

# 02. Get the list of groups for which to pull
# gene lists (up- and down-regulated DEGs) and
# run GO analysis
grouping_values = [
        _.strip()
        for _ in args.group_to_process.split('[')[
            -1].split(']')[
                0].split(',')
            ]
if grouping_values == 'None':
    grouping_values = [
            _ for _ in os.listdir(dea_results_dir)
            if os.path.isdir(f'{dea_results_dir}/{_}')
            ]
gv_str = '\n\t'.join(grouping_values)
print(f'Groups for which DEGs will be compared:\n\t{gv_str}')

# 04. Iterate through the specified test condition comparisons
# for each of the specified groups, find the union of all DEGs
# across all test conditions, and generate a visualization
# showing their overlaps and expression values
for group in grouping_values:
    # Set a variable to hold the comparison type
    ov_pairs_tc = group_ov_pair_dict[group]
    print(f'\n\nComparing DEG lists for {group}')
    gene_list_dir = f'{dea_results_dir}/{group}'
    group_results_dir = f'{dea_comp_out_dir}/{group}'
    if not os.path.isdir(group_results_dir):
        os.system(f'mkdir -p {group_results_dir}')

    # 04a. Read in the up- and down- regulated DEG
    # lists for the specified comparisons
    up_deg_dict = {}
    down_deg_dict = {}
    up_deg_df = pd.DataFrame()
    down_deg_df = pd.DataFrame()
    for tc in test_comparisons:
        tc_key = '_'.join(tc)
        print(f'Finding deg files for {tc_key}:')
        # 04a.i. Pull upregulated DEGs
        up_deg_fn = [
                _ for _ in
                os.listdir(gene_list_dir)
                if
                all(
                    [
                        tag in _
                        for tag in
                        tc + [gene_df_upreg_root]
                        ]
                    )
                ]
        if len(up_deg_fn)>0:
            up_deg_fn = up_deg_fn[0]
            up_full_fn = f'{gene_list_dir}/{up_deg_fn}'
            df_curr = pd.read_csv(
                    up_full_fn,
                    index_col=0
                    )
            df_curr['test_cond'] = [tc_key]*len(df_curr)
            pcols = [
                    _ for _ in
                    df_curr.columns.values.tolist()
                    if dea_p_col_tag in _
                    ]
            cols = ['test_cond', l2fc_col] + pcols
            if len(up_deg_df)==0:
                up_deg_df = df_curr[cols].copy()
            else:
                up_deg_df = pd.concat(
                        [
                            up_deg_df,
                            df_curr[cols].copy()
                            ],
                        ignore_index=False
                        )
            deg_list_curr = df_curr.index.values.tolist()
            if len(deg_list_curr) > 0:
                up_deg_dict[tc_key] = deg_list_curr
            del df_curr
        # 04a.ii. Pull downregulated DEGs
        down_deg_fn = [
                _ for _ in
                os.listdir(gene_list_dir)
                if
                all(
                    [
                        tag in _
                        for tag in
                        tc + [gene_df_downreg_root]
                        ]
                    )
                ]
        if len(down_deg_fn)>0:
            down_deg_fn = down_deg_fn[0]
            down_full_fn = f'{gene_list_dir}/{down_deg_fn}'
            df_curr = pd.read_csv(
                    down_full_fn,
                    index_col=0
                    )
            df_curr['test_cond'] = [tc_key]*len(df_curr)
            pcols = [
                    _ for _ in
                    df_curr.columns.values.tolist()
                    if dea_p_col_tag in _
                    ]
            cols = ['test_cond', l2fc_col] + pcols
            if len(up_deg_df)==0:
                down_deg_df = df_curr[cols].copy()
            else:
                down_deg_df = pd.concat(
                        [
                            down_deg_df,
                            df_curr[cols].copy()
                            ],
                        ignore_index=False
                        )
            deg_list_curr = df_curr.index.values.tolist()
            if len(deg_list_curr) > 0:
                down_deg_dict[tc_key] = deg_list_curr

    # 04b. Get the union of all DEGs across comparisons
    tc_all_degs = []
    for key, value in up_deg_dict.items():
        tc_all_degs.extend(value)
    for key, value in down_deg_dict.items():
        tc_all_degs.extend(value)
    tc_all_degs = list(set(tc_all_degs))

    # 04c. Join up- and down- regulated DEG DataFrames
    # for easier manipulation with visualizations
    up_deg_df['direction'] = ['up']*len(up_deg_df)
    down_deg_df['direction'] = ['down']*len(down_deg_df)
    all_deg_df = pd.concat(
            [
                up_deg_df,
                down_deg_df
                ],
            ignore_index=False
            )

    # 04d. Order the up- and down- DEGs so that the upregulated DEGs
    # for the first test condition of interest are in a block
    all_up_degs = list(set(up_deg_df.index.values.tolist()))
    all_down_degs = list(set(down_deg_df.index.values.tolist()))
    down_degs_to_add = list(set(all_down_degs) - set(all_up_degs))
    all_degs_ordered = all_up_degs + down_degs_to_add

    # 04e. Reset the index of the all-DEG DataFrame, so that
    # you can identify rows of interest more easily
    all_deg_df.reset_index(
            drop=False,
            inplace=True
            )

    # 04f. Build a heat map for the DEGs across conditions,
    # following the test condition structure, which organizes
    # test conditions into supergroups
    # 04f.i. Define a matrix with test conditions as rows
    # and DEGs as columns
    degs_mtx = np.zeros(
            (
                len(test_comparisons),
                len(tc_all_degs)
                )
            )
    row_idx = 0
    test_cond_strs = []
    test_cond_vals = []
    test_cond_to_row_idx_dict = {}
    to_draw_rectangle = False
    for sg_key, sg_tcs in tc_subset_mapping.items():
        for sg_tc in sg_tcs:
            tc_curr = '_'.join(sg_tc)
            case_curr_pieces = sg_tc[1].split('_')
            ctrl_curr_pieces = sg_tc[2].split('_')
            case_hiv = HIV_map['_'.join(case_curr_pieces[:-1])]
            ctrl_hiv = HIV_map['_'.join(ctrl_curr_pieces[:-1])]
            case_sud = SUD_map[case_curr_pieces[-1]]
            ctrl_sud = SUD_map[ctrl_curr_pieces[-1]]
            tc_curr_format = ''
            if case_sud == ctrl_sud:
                # is an HIV comparison
                tc_curr_format = f'{case_hiv} vs. {ctrl_hiv},\n{case_sud}'
                to_draw_rectangle=True
            else:
                # is an SUD comparison
                tc_curr_format = f'{case_sud} vs. {ctrl_sud},\n{case_hiv}'
            test_cond_strs.append(tc_curr_format)
            test_cond_vals.append(tc_curr)
            test_cond_to_row_idx_dict[tc_curr] = row_idx
            for deg_idx,deg_curr in enumerate(all_degs_ordered):
                deg_val_row = all_deg_df.loc[
                        (
                            (all_deg_df['gene_name']==deg_curr)
                            &
                            (all_deg_df['test_cond']==tc_curr)
                            )
                        ]
                deg_val = None
                if len(deg_val_row) > 0:
                    deg_val = deg_val_row[l2fc_col].values.tolist()[0]
                else:
                    deg_val = np.nan
                degs_mtx[row_idx,deg_idx] = deg_val
            row_idx += 1

    # 04g. Get the DEG overlaps between conditions
    # of interest in a manner based on comparison type
    # 04g.i. Set up an overlap dictionary
    overlap_dict = {}
    # 04g.ii. Get the pairs of test conditions
    # for which to compute overlaps
    overlap_pairs = []
    for ovptc in ov_pairs_tc:
        ovptc_pieces = ovptc.split(',')
        ovp_curr = (
                test_cond_to_row_idx_dict[ovptc_pieces[0]],
                test_cond_to_row_idx_dict[ovptc_pieces[1]],
                )
        overlap_pairs.append(ovp_curr)
    # 04g.iii. Compute overlaps
    for op in overlap_pairs:
        ud_dict_curr = {}
        # 04g.iii.A. Get the test
        # conditions to compare
        tc1 = test_cond_vals[op[0]]
        tc2 = test_cond_vals[op[1]]
        # 04g.iii.B. Pull upregulated DEGs
        # for both conditions and compare
        up_tc1_degs = all_deg_df.loc[
                (
                    (all_deg_df['test_cond']==tc1)
                    &
                    (all_deg_df['direction']=='up')
                    )
                ]['gene_name'].values.tolist()
        up_tc2_degs = all_deg_df.loc[
                (
                    (all_deg_df['test_cond']==tc2)
                    &
                    (all_deg_df['direction']=='up')
                    )
                ]['gene_name'].values.tolist()
        up_ovp = list(set(up_tc1_degs).intersection(set(up_tc2_degs)))
        ud_dict_curr['up'] = up_ovp
        # 04g.iii.C. Pull downregulated
        # DEGs for both conditions and compare
        down_tc1_degs = all_deg_df.loc[
                (
                    (all_deg_df['test_cond']==tc1)
                    &
                    (all_deg_df['direction']=='down')
                    )
                ]['gene_name'].values.tolist()
        down_tc2_degs = all_deg_df.loc[
                (
                    (all_deg_df['test_cond']==tc2)
                    &
                    (all_deg_df['direction']=='down')
                    )
                ]['gene_name'].values.tolist()
        down_ovp = list(set(down_tc1_degs).intersection(set(down_tc2_degs)))
        ud_dict_curr['down'] = down_ovp
        # 04g.iii.D. Store overlaps for current test condition pair
        overlap_dict[op] = ud_dict_curr


    # 04h. Plot the matrix as a heatmap
    # 04h.i. Define as axis object
    fig,ax = plt.subplots()
    mtx_aspect_ratio = degs_mtx.shape[1]*1.0/degs_mtx.shape[0]
    w_in = 3
    h_in = 1
    figw,figh = set_size(w_in,h_in)
    pad_size = 0.01
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.subplots_adjust(
            left = pad_size,
            right = 1-pad_size,
            top = 1-pad_size,
            bottom = pad_size
            )
    # 04h.ii. Specify heatmap extent in data coordinates
    # as (left, right, top, bottom)
    row_factor = 40
    col_factor = 4
    col_length = degs_mtx.shape[1]*col_factor
    if col_length > 3000:
        row_factor *= 4

    row_length = degs_mtx.shape[0]*row_factor
    print(f'Col length: {col_length}\nRow length: {row_length}')

    extent_curr = [
            0,
            col_length,
            0,
            row_length
            ]
    # 04h.iii. Set up a colormap that includes a nan color
    cmap_nan = 'darkgray'
    cmap_color_list = ['midnightblue', 'blue', 'dodgerblue', 'lightskyblue', 'white', 'lightpink', 'red', 'firebrick', 'darkred']
    cmap_node_list = [0.0, 0.125, 0.25, 0.499, 0.5, 0.501, 0.75, 0.875, 1.0]
    cmap_curr = LinearSegmentedColormap.from_list('br_no_white',list(zip(cmap_node_list,cmap_color_list)))
    cmap_curr.set_bad(color=cmap_nan)
    l2fcs = all_deg_df.cross_fold_l2fc_mean.values.tolist()
    if len(l2fcs) > 0:
        lim = np.ceil(np.max([np.abs(np.min(l2fcs)),np.abs(np.max(l2fcs))]))
        # 04h.iv. Define a dictionary of colorbar
        # keywords
        tick_list = [_ for _ in np.arange(-lim,lim+0.01,1.0)]
        cb_kw = {
                'shrink':0.8,
                'aspect':8,
                'orientation':'vertical',
                'ticks':tick_list
                }
        heatmap(
                degs_mtx,
                row_labels=test_cond_strs,
                col_labels=['']*degs_mtx.shape[1],
                ax=ax,
                cbar_kw=cb_kw,
                cbarlabel='log2fc',
                cmap=cmap_curr,
                vmin=-lim,
                vmax=lim,
                origin='lower',
                extent=extent_curr,
                interpolation='none'
                )
        # 04h.v. Adjust axis ticks
        ax.set_xticks(
                np.arange(0,extent_curr[1]+1,col_factor),
                minor=True)#-.5*col_factor, minor=True)
        ax.set_yticks(
                np.arange(0,extent_curr[3]+1,row_factor),
                minor=True)#-.5*row_factor, minor=True)
        # 04h.vi. Adjust axis tick labels
        ax.set_yticks(
                np.arange(0,extent_curr[3],row_factor) + .5*row_factor,
                labels=test_cond_strs
                )
        # 04h.vii. Draw a rectangle around SUD+ cases
        # if this is an HIV comparison plot
        if to_draw_rectangle==True:
            p=plt.Rectangle(
                    (0,0.5*extent_curr[3]),
                    width=extent_curr[1],
                    height=0.5*extent_curr[3],
                    fill=False,
                    color='green'
                    )
            p.set_clip_on(False)
            p.set_snap(True)
            p.set_alpha(1)
            p.set_zorder(6)
            ax.add_patch(p)

        # 04h.viii. Specify and make an output directory
        # and save a copy of the figure without any labels
        # summarizing overlaps
        out_dir = f'{dea_comp_out_dir}/{group}'
        if not os.path.isdir(out_dir):
            os.system(f'mkdir -p {out_dir}')
        out_no_lbls_full_fn = f'{out_dir}/{group}_deg_levels_across_comparison_no_labels.png'
        plt.savefig(
                out_no_lbls_full_fn,
                dpi=600,
                bbox_inches='tight'
                )

        # 04h.viii. Plot brackets and text to show the degree
        # of each pair of overlaps
        for op,ud_dict in overlap_dict.items():
            # 04h.viii.A. Get vertical positions of bracket
            # edges from pair locations
            p1 = op[0]
            p2 = op[1]
            dp = np.abs(p2-p1)
            yp1 = (p1+0.5)*row_factor+dp
            yp2 = (p2+0.5)*row_factor-dp
            # 04h.viii.B. Get horizontal positions of bracket
            # edges from difference between pair locations
            xp1 = 1.0
            xp2 = xp1 + 0.01 + 0.02*(dp-1)*p2
            # 04h.viii.C. Set up share string as label
            up_share = ud_dict['up']
            down_share = ud_dict['down']
            n_up_share = len(up_share)
            n_down_share = len(down_share)
            up_text = f'\u2191{n_up_share}\n'
            down_text = f'\u2193{n_down_share}'
            txt_rot = 'horizontal'
            txt_space_factor = int(np.max([1,np.ceil(2*extent_curr[1]/1000.0)]))

            # 04h.viii.D. Make vertical brackets
            hb,ht = make_brackets(
                    ax,
                    is_horiz=False,
                    pos1=yp1,
                    pos2=yp2,
                    label=[up_text,down_text], #share_text,
                    label_color=['firebrick','navy'],
                    spine_pos=xp2,
                    tip_pos=xp1,
                    txt_rot=txt_rot,
                    txt_space_factor=txt_space_factor
                    )
        out_full_fn = f'{out_dir}/{group}_deg_levels_across_comparison.png'
        plt.savefig(
                out_full_fn,
                dpi=600,
                bbox_inches='tight'
                )
        plt.close('all')

sys.exit()   
