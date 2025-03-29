import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
import yaml

# This script take the results of DEA for the specified
# groups/cell types and test condition level comparisons
# and does the following for each:
# (1) identifies differentially expressed genes (DEGs)
#     using k-fold cross-validation
# (2) saves individual DEG lists to file
# (3) generates DEG volcano plots showing log2 fold
#     change in test vs. control mean expression
#     vs. mean normed count per gene
# (4) produces volcano plots with specified DEGs
#     of interest plotted in a different color.

# Script setup

# 00. Create argparse objects to read input arguments
parser = argparse.ArgumentParser(description='Visualization of differentially expressed genes' + \
        ' for selected expression data partitions.')
# 00a. Configuration file parameters
parser.add_argument(f'--config-yaml-path',type=str,required=True)
# 00b. Groups/cell types to process
parser.add_argument(f'--group-to-process',type=str,required=True)
args = parser.parse_args()

# 00c. Set plotting parameters
# 00c.i. Set font size for plots
plt.rcParams['font.size']=10
# 00c.ii. Set marker size for plots
MARKERSIZE=0.8

# 01. Get analysis parameters from configuration file
cfg = None
with open(args.config_yaml_path) as f:
    cfg = yaml.safe_load(f)
# 01a. Dataset information
# 01a.i. Data root directory
qc_root_dir = cfg.get('root_dir')
# 01a.ii. DEA results input directory
dea_data_dir = qc_root_dir + f'/' + cfg.get('dea_data_dir')
# 01a.iii. DEA results output directory
output_dea_data_dir = qc_root_dir + f'/' + cfg.get('output_dea_data_dir')
# 01a.iv. Results directories
dea_diffexpr_dir = dea_data_dir + f'/' + cfg.get('dea_deseq_formatted_data_dir')
dea_results_dir = dea_diffexpr_dir + f'/' + cfg.get('dea_results_dir')
# 01a.v. Output results directories
output_dea_diffexpr_dir = output_dea_data_dir + f'/' + cfg.get('dea_deseq_formatted_data_dir')
output_dea_results_dir = output_dea_diffexpr_dir + f'/' + cfg.get('dea_results_dir')
# 01a.iv. Optional subdirectory for output
# This variable will only be read and used if it
# is in the configuration file (it is currently 
# only in the configuration file for SUDOPCN
# analysis with alternative alpha value for DEA)
dea_results_subdir = None
try:
    dea_results_subdir = f'/' + cfg.get('dea_results_sub_dir')
except:
    dea_results_subdir = ''
if dea_results_subdir is None:
    dea_results_subdir = ''

# 01b. DEA parameters for visualization
# 01b.i. DEG detection significance level
alpha = cfg.get('dea_alpha') #0.1
# 01b.ii. Cell type label used for partitioning
grouping = cfg.get('dea_partition_variable')
# 01b.iii. Test condition comparisons that
# would have been tested if powered
test_comparisons = cfg.get('dea_cond_comparison_list')
# 01b.iv. Number of folds used for k-fold cross-validation
N_FOLDS_KFCV = cfg.get('dea_n_folds_kfcv')
# 01b.v. Top number of over-/under- expressed genes
# to report in more restricted lists
N_TOP = cfg.get('dea_n_top')
# 01b.vi. Test type (for generating results file names)
TEST_TYPE = cfg.get('dea_test_type')

# 02. Read in results files for individual clusters
# 02a. Get the list of groups with results
grouping_values = args.group_to_process
if grouping_values == 'None':
    grouping_values = [
            _ for _ in os.listdir(dea_results_dir)
            if os.path.isdir(f'{dea_results_dir}/{_}')
            ]
else:
    grouping_values = [
            _.strip()
            for _ in
            grouping_values.split(',')
            ]

gv_str = '\n'.join(grouping_values)
print(f'Grouping values to be processed:\n{gv_str}')

for g in grouping_values:
    # 02b. Import all Wald test results for all comparisons
    # performed for this group
    # Also import the normalized counts DataFrame
    print(f'Processing results for {grouping} group {g}...')
    dea_results_group_dir = f'{dea_results_dir}/{g}'
    output_dea_results_group_dir = f'{output_dea_results_dir}/{g}'
    dea_results_group_out_dir = f'{output_dea_results_group_dir}{dea_results_subdir}'
    if not os.path.isdir(dea_results_group_out_dir):
        os.system(f'mkdir -p {dea_results_group_out_dir}')
    files = os.listdir(dea_results_group_dir)
    test_type_str = TEST_TYPE.lower()
    #results_files_all = [_ for _ in files if 'wald_results' in _]
    results_files_all = [_ for _ in files if f'{test_type_str}_results' in _]
    #norm_count_files_all = [_ for _ in files if 'wald_norm_counts' in _]
    norm_count_files_all = [_ for _ in files if f'{test_type_str}_norm_counts' in _]

    # 02c. Iterate through each comparison in the list
    for tc_curr in test_comparisons:
        tc_str = f'{tc_curr[0]}: {tc_curr[1]} vs. {tc_curr[2]}'
        tc_curr_plt_str = '_'.join(tc_curr)
        print(f'Processing results for\n\t{tc_str}')
        # 02c.i. In order to prevent unintended conditions
        # with embedded matches from being included in the
        # results files list, we append an underscore
        # to the end of each test comparison component
        # since DEA results files are named by joining
        # the test condition, treatment condition level,
        # and reference condition level by underscores.
        tc_curr_fn_tags = [
                f'{_}_'
                for _ in
                tc_curr]
        results_files = [
                _ for _ in results_files_all
                if
                (
                    (tc_curr_fn_tags[0] in _)
                    &
                    (tc_curr_fn_tags[1] in _)
                    &
                    (tc_curr_fn_tags[2] in _)
                    )
                ]
        norm_count_files = [
                _ for _ in norm_count_files_all
                if
                (
                    (tc_curr_fn_tags[0] in _)
                    &
                    (tc_curr_fn_tags[1] in _)
                    &
                    (tc_curr_fn_tags[2] in _)
                    )
                ]

        rf_str = '\n\t\t'.join(results_files)
        ncf_str = '\n\t\t'.join(norm_count_files)
        print(f'\tResults files:\n\t\t{rf_str}')
        print(f'\n\tNormed count files:\n\t\t{ncf_str}')

        # 02d. Only proceed if the current test condition
        # comparison was powered and has results
        # files
        if (
                (len(results_files) > 0)
                &
                (len(norm_count_files) > 0)
                ):
            # 03. Merge results across folds for the
            # current comparison
            res_merged_df = pd.DataFrame()

            # 03a. Get the up- and down-regulated genes
            # for each fold-removed subset
            folds = [f'{_}' for _ in np.arange(1,N_FOLDS_KFCV+1)]
            for fold in folds:
                print(f'Processing fold {fold} results for this comparison...')
                fold_str = f'fold_{fold}'
                results_file = [_ for _ in results_files if fold_str in _][0]
                norm = [_ for _ in norm_count_files if fold_str in _][0]
                res_df = pd.read_csv(f'{dea_results_group_dir}/{results_file}',index_col=0)
                norm_df = pd.read_csv(f'{dea_results_group_dir}/{norm}',index_col=0)

                # 03a.i. Make sure the indices for the results and normed counts DataFrames match
                res_idx = res_df.index.values.tolist()
                norm_idx = norm_df.index.values.tolist()
                idxs_consistent = (res_idx == norm_idx)
                if idxs_consistent:
                    print(f'Result and normalized count indices match. Proceeding...')
                else:
                    print(f'Result and normalized count indices do not match.')
                    break

                # 03a.ii. Compute the mean normalized count value for each gene in the results file
                cols_for_mean = [_ for _ in norm_df.columns.tolist() if 'gene_name' not in _]
                norm_df['norm_count_mean'] = np.mean(norm_df[cols_for_mean].to_numpy(),axis=1)
                norm_mean_df = norm_df['norm_count_mean'].copy()
                res_df = res_df.join(norm_mean_df)

                # 03a.iii. Delete the normed counts DataFrame to save space
                del norm_df

                # 03a.iv. Merge the data
                # 03a.iv.a. Add fold information to the names
                # of columns from the current results DataFrame
                col_map = {}
                for col in res_df:
                    col_map[col] = f'{col}_{fold}'
                res_df.rename(
                        columns=col_map,
                        inplace=True
                        )
                if len(res_merged_df) == 0:
                    res_merged_df = res_df.copy()
                else:
                    res_merged_df = res_merged_df.join(
                            res_df,
                            how='outer',
                            rsuffix=f'_{fold}'
                            )
                del res_df

            print(f'Cross-fold results DataFrame:')
            print(res_merged_df.head(),'\n')
            
            # 03b. Compute the mean and standard deviation of the across-fold
            # expression levels and log2fold changes
            l2fc_cols = [
                    _ for _ in res_merged_df.columns.values.tolist()
                    if 'log2FoldChange' in _]
            exp_cols = [
                    _ for _ in res_merged_df.columns.values.tolist()
                    if 'norm_count_mean' in _
                    ]
            res_merged_df['cross_fold_norm_count_mean'] = np.mean(res_merged_df[exp_cols].to_numpy(),axis=1)
            res_merged_df['cross_fold_norm_count_std'] = np.std(res_merged_df[exp_cols].to_numpy(),axis=1)
            res_merged_df['cross_fold_l2fc_mean'] = np.mean(res_merged_df[l2fc_cols].to_numpy(),axis=1)
            res_merged_df['cross_fold_l2fc_std'] = np.std(res_merged_df[l2fc_cols].to_numpy(),axis=1)

            # 03c. Identify genes with significant log2 fold changes across all folds
            padj_cols = [
                    _ for _ in res_merged_df.columns.values.tolist()
                    if 'padj' in _
                    ]
            res_merged_sig_idx = res_merged_df.loc[
                    (res_merged_df[padj_cols] < alpha).all(axis=1)
                    ].index
            res_merged_not_sig_idx = list(
                    set(res_merged_df.index.values.tolist()) - set(res_merged_sig_idx)
                    )
            # 03c.i. Pull a copy of the significantly up- or down-regulated genes for plotting
            # and separated saving
            res_merged_sig = res_merged_df.loc[res_merged_sig_idx].copy()
            # 03c.ii. Add an in-line column with significance flag and save the full set
            # of genes, sorted by their mean log2 fold changes, to file
            res_merged_df['cross_fold_sig'] = 0
            res_merged_df.loc[res_merged_sig_idx,'cross_fold_sig'] = 1
            
            # 03d. Save the per- and cross-fold log2 fold change expression levels
            all_fn = f'{dea_results_group_out_dir}/diff_expr_levels_all__{tc_curr_plt_str}.csv'
            res_merged_df_to_save = res_merged_df.copy()
            res_merged_df_to_save.index.rename('gene_name_old',
                    inplace=True)
            res_merged_df_to_save['gene_name'] = [_.split('GRCh38_______________')[-1]
                    for _ in res_merged_df_to_save.index.values.tolist()]
            res_merged_df_to_save.set_index('gene_name',
                    drop=True,
                    inplace=True)
            res_merged_df_to_save.to_csv(
                    all_fn,
                    index=True
                    )

            print(f'Saved cross-fold expression levels, log2 fold changes, and p-values to file.')
            del res_merged_df_to_save

            # 03e. Pull separate lists of significantly up- and downregulated genes
            upreg = res_merged_sig.loc[
                    res_merged_sig['cross_fold_l2fc_mean'] > 0
                    ]
            downreg = res_merged_sig.loc[
                    res_merged_sig['cross_fold_l2fc_mean'] < 0
                    ]
            print(f'Genes with padj < {alpha}: {len(res_merged_sig)} ({len(res_merged_sig)*100.0/len(res_merged_df):0.002}% high-count genes)')
            print(f'\t Upregulated: {len(upreg)} ({len(upreg)*100.0/len(res_merged_df):0.002}%)')
            print(f'\t Downregulated: {len(downreg)} ({len(downreg)*100.0/len(res_merged_df):0.002}%)')

            # 03f. Plot the cross-fold log2 fold change as a function of mean normed count
            # and color markers by whether the change in expression is significant
            # at the specified alpha level
            # 03f.i. Set the desired plot
            # width and height in inches
            fig_w = 1.6
            fig_h = 1.6
            fig,ax = plt.subplots(
                    figsize=(fig_w,fig_h),
                    layout='constrained'
                    )
            plt.plot(res_merged_df.loc[res_merged_not_sig_idx]['cross_fold_norm_count_mean'].values.tolist(),
                    res_merged_df.loc[res_merged_not_sig_idx]['cross_fold_l2fc_mean'].values.tolist(),
                    markersize=MARKERSIZE,
                    marker='o',
                    linestyle='None',
                    color='gray')
            # 03f.ii. Plot data for genes with significant lfc in the range (-2,2)
            res_sig_lfc_btw_neg_2_2 = res_merged_sig.loc[                           
                    list(
                        set(res_merged_sig.loc[res_merged_sig['cross_fold_l2fc_mean'] > -2].index.values).intersection(
                            set(res_merged_sig.loc[res_merged_sig['cross_fold_l2fc_mean'] < 2].index.values))
                        )
                    ].copy()
            res_sig_lfc_btw_neg_2_2_upreg = res_sig_lfc_btw_neg_2_2.loc[
                    res_sig_lfc_btw_neg_2_2['cross_fold_l2fc_mean'] > 0
                    ].copy()
            res_sig_lfc_btw_neg_2_2_downreg = res_sig_lfc_btw_neg_2_2.loc[
                    res_sig_lfc_btw_neg_2_2['cross_fold_l2fc_mean'] < 0
                    ].copy()
            plt.plot(res_sig_lfc_btw_neg_2_2_upreg['cross_fold_norm_count_mean'].values.tolist(),
                    res_sig_lfc_btw_neg_2_2_upreg['cross_fold_l2fc_mean'].values.tolist(),
                    markersize=MARKERSIZE,
                    marker='o',
                    linestyle='None',
                    color='red')
            plt.plot(res_sig_lfc_btw_neg_2_2_downreg['cross_fold_norm_count_mean'].values.tolist(),
                    res_sig_lfc_btw_neg_2_2_downreg['cross_fold_l2fc_mean'].values.tolist(),
                    markersize=MARKERSIZE,
                    marker='o',
                    linestyle='None',
                    color='blue')
            res_lfc_gt_2 = res_merged_sig.loc[res_merged_sig['cross_fold_l2fc_mean'] > 2].copy()
            res_lfc_lt_neg2 = res_merged_sig.loc[res_merged_sig['cross_fold_l2fc_mean'] < -2].copy()
            plt.plot(res_lfc_gt_2['cross_fold_norm_count_mean'].values.tolist(),
                    res_lfc_gt_2['cross_fold_l2fc_mean'].values.tolist(),
                    markersize=MARKERSIZE,
                    marker='^',
                    linestyle='None',
                    color='red')
            plt.plot(res_lfc_lt_neg2['cross_fold_norm_count_mean'].values.tolist(),
                    res_lfc_lt_neg2['cross_fold_l2fc_mean'].values.tolist(),
                    markersize=MARKERSIZE,
                    marker='v',
                    linestyle='None',
                    color='blue')
            # 03f.iii. Plot visual guide lines at 1/2- and 2-fold change marks
            plt.plot([0,500],
                    [-1.0,-1.0],
                    '--',
                    linewidth=1,
                    color='gray',
                    alpha=0.6)
            plt.plot([0,500],
                    [1.0,1.0],
                    '--',
                    linewidth=1,
                    color='gray',
                    alpha=0.6)
            # 03f.iv. Adjust axis limits to accommodate
            # all data points
            plt.xlim((0.95,100))
            plt.xscale('log')
            plt.ylim((-2.5,2.5))
            # 03f.v. Adjust axis tick properties
            plt.xticks(fontsize=6)
            yt_curr = [
                    _ for _ in
                    ax.get_yticks()
                    if np.abs(_)<2.5
                    ]
            ytl_curr = [
                    f'{_:.0f}'
                    for _ in
                    yt_curr
                    ]
            ax.set_yticks(
                    yt_curr,
                    labels=ytl_curr,
                    fontsize=6
                    )
            plt.xlabel('mean normalized count')
            plt.ylabel('log2 fold change')
            plt.title(tc_str)
            plt.savefig(
                    f'{dea_results_group_out_dir}/cross_fold_lfc_raw_vs_mean_norm_counts__{tc_curr_plt_str}.png',
                    dpi=500,
                    bbox_inches='tight'
                    )
            # 03f.v. Save a variant on this plot with
            # marker genes of interest, the dopamine
            # and monoamine reuptake transporters SLC6A3 and
            # SLC18A2, respectively, plotted in different colors
            # if they are DEGs
            save_new_plot = False
            if 'GRCh38_______________SLC6A3' in res_merged_sig.index:
                plt.plot(
                        res_merged_sig.loc['GRCh38_______________SLC6A3']['cross_fold_norm_count_mean'],
                        res_merged_sig.loc['GRCh38_______________SLC6A3']['cross_fold_l2fc_mean'],
                        markersize=MARKERSIZE,
                        marker='o',
                        linestyle='None',
                        color='blueviolet'
                        )
                save_new_plot = True
            if 'GRCh38_______________SLC18A2' in res_merged_sig.index:
                plt.plot(
                        res_merged_sig.loc['GRCh38_______________SLC18A2']['cross_fold_norm_count_mean'],
                        res_merged_sig.loc['GRCh38_______________SLC18A2']['cross_fold_l2fc_mean'],
                        markersize=MARKERSIZE,
                        marker='o',
                        linestyle='None',
                        color='magenta'
                        )
                save_new_plot = True
            if save_new_plot==True:
                plt.savefig(
                        f'{dea_results_group_out_dir}/cf_l2fc_raw_vs_mean_norm_counts__{tc_curr_plt_str}_w_DEGs_SLC6A3_bluviolet_SLC18A2_magenta.png',
                        dpi=500
                        )
            plt.close('all')
            print(f'Sideways volcano plot saved.')
            print(f'\n')

            # 03g. Sort the significantly overexpressed genes in the current
            # group in descending order
            upreg.sort_values(by=['cross_fold_l2fc_mean'],
                    ascending=False,
                    inplace=True)
            # 03h. Sort significantly downexpressed genes in ascending order so the genes with
            # largest magnitudes of under-expression come first
            downreg.sort_values(by=['cross_fold_l2fc_mean'],
                    ascending=True,
                    inplace=True)
            
            # 03i. Plot the distribution of log2 fold changes in genes
            # that are significantly up- or down-regulated
            plt.figure()
            plt.hist(upreg['cross_fold_l2fc_mean'].values.tolist(),
                    alpha=0.8)
            plt.xlabel('mean log2 fold change')
            plt.ylabel('frequency')
            plt.title(f'Upregulated DEG log2 fold change,\n{tc_str}')
            plt.savefig(f'{dea_results_group_out_dir}/sig_overexpr_level_hist__alpha_{alpha}__{tc_curr_plt_str}.png')
            plt.close()
            
            plt.figure()
            plt.hist(downreg['cross_fold_l2fc_mean'].values.tolist(),
                    alpha=0.8)
            plt.xlabel('mean log2 fold change')
            plt.ylabel('frequency')
            plt.title(f'Downregulated DEG log2 fold change,\n{tc_str}')
            plt.savefig(f'{dea_results_group_out_dir}/sig_underexpr_level_hist__alpha_{alpha}__{tc_curr_plt_str}.png')
            plt.close()
            print(f'DEG expression level plots saved.')
            
            # 03j. Save both the full lists of over- and under-expressed genes
            # and the lists of the top-n genes in both, keeping the truncated gene
            # name, the log2 fold change, and the adjusted p-value.
            print(f'Saving significantly up- and down-regulated DEGs...')
            # 03j.i. Get the list of per-fold and across-fold columns to save
            save_cols = ['cross_fold_l2fc_mean','cross_fold_l2fc_std'] + l2fc_cols + padj_cols 
            # 03j.ii. Generate upregulated gene lists
            upreg_to_save = upreg[save_cols].copy()
            upreg_to_save.index.rename('gene_name_old',
                    inplace=True)
            upreg_to_save['gene_name'] = [_.split('GRCh38_______________')[-1]
                    for _ in upreg_to_save.index.values.tolist()]
            upreg_to_save.set_index('gene_name',
                    drop=True,
                    inplace=True)
            upreg_to_save_topn = upreg_to_save.head(N_TOP).copy()
            del upreg
            # 03j.iii. Generate downregulated gene lists
            downreg_to_save = downreg[save_cols].copy()
            downreg_to_save.index.rename('gene_name_old',
                    inplace=True)
            downreg_to_save['gene_name'] = [_.split('GRCh38_______________')[-1]
                    for _ in downreg_to_save.index.values.tolist()]
            downreg_to_save.set_index('gene_name',
                    drop=True,
                    inplace=True)
            downreg_to_save_topn = downreg_to_save.head(N_TOP).copy()
            del downreg
            # 03j.iv. Save upregulated gene lists
            upreg_fn = f'{dea_results_group_out_dir}/sig_overexpr_genes_all__alpha_{alpha}__{tc_curr_plt_str}.csv'
            upreg_topn_fn = f'{dea_results_group_out_dir}/sig_overexpr_genes_top{N_TOP}__alpha_{alpha}__{tc_curr_plt_str}.csv'
            upreg_to_save.to_csv(upreg_fn,
                    index=True)
            upreg_to_save_topn.to_csv(upreg_topn_fn,
                    index=True)
            # 03j.v. Save downregulated gene lists
            downreg_fn = f'{dea_results_group_out_dir}/sig_underexpr_genes_all__alpha_{alpha}__{tc_curr_plt_str}.csv'
            downreg_topn_fn = f'{dea_results_group_out_dir}/sig_underexpr_genes_top{N_TOP}__alpha_{alpha}__{tc_curr_plt_str}.csv'
            downreg_to_save.to_csv(downreg_fn,
                    index=True)
            downreg_to_save_topn.to_csv(downreg_topn_fn,
                    index=True)
            print(f'\n')

        else:
            print(f'No results files were found for this group; skipping.\n')


sys.exit()
