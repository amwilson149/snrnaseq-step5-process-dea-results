import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
import yaml

# This script take the results of the diffexpr analysis
# (a Python wrapper for DESeq2)
# and produces visualizations of the results.

# Script setup

# 00. Create an argparse object for reading input arguments
parser = argparse.ArgumentParser(description='Visualization of differentially expressed genes' + \
        ' for selected expression data partitions.')
parser.add_argument(f'--config-yaml-path',type=str,required=True)
parser.add_argument(f'--group-to-process',type=str,required=True)
args = parser.parse_args()

# 01. Get analysis parameters from configuration file
cfg = None
with open(args.config_yaml_path) as f:
    cfg = yaml.safe_load(f)
# 01a. Dataset information
# 01a.i. Data root directory
qc_root_dir = cfg.get('root_dir')
# 01a.ii. Differential expression analysis output directory
dea_data_dir = qc_root_dir + f'/' + f'/' + cfg.get('dea_data_dir')
dea_diffexpr_dir = dea_data_dir + f'/' + cfg.get('dea_deseq_formatted_data_dir')
dea_results_dir = dea_diffexpr_dir + f'/' + cfg.get('dea_results_dir')
# 01a.iv. Optional subdirectory for output
dea_results_subdir = None
try:
    dea_results_subdir = f'/' + cfg.get('dea_results_sub_dir')
except:
    dea_results_subdir = ''


# 01b. DEA parameters for visualization
# 01b.i. DEG detection significance level
alpha = cfg.get('dea_alpha')
# 01b.ii. Cell type label used for partitioning
grouping = cfg.get('dea_partition_variable')
# 01b.iii. Test condition comparisons that 
# would have been tested if powered
test_comparisons = cfg.get('dea_cond_comparison_list')
# 01b.iv. Number of folds used for k-fold cross-validation
N_FOLDS_KFCV = cfg.get('dea_n_folds_kfcv')

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
    print(f'Searching for files in {dea_results_group_dir}....')
    dea_results_group_out_dir = f'{dea_results_group_dir}{dea_results_subdir}'
    if not os.path.isdir(dea_results_group_out_dir):
        os.system(f'mkdir -p {dea_results_group_out_dir}')
    files = os.listdir(dea_results_group_dir)
    results_files_all = [_ for _ in files if 'wald_results' in _]
    norm_count_files_all = [_ for _ in files if 'wald_norm_counts' in _]

    # 02c. Iterate through each comparison in the list
    for tc_curr in test_comparisons:
        tc_curr_str = f'{tc_curr[0]}: {tc_curr[1]} vs. {tc_curr[2]}'
        tc_curr_plt_str = '_'.join(tc_curr)
        print(f'Processing results for\n\t{tc_curr_str}')
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
        print(f'\n\tNormed counts files:\n\t\t{ncf_str}')

        # 02d. Only proceed if the current test condition
        # comparison was powered and has results files
        if (
                (len(results_files) > 0)
                &
                (len(norm_count_files) > 0)
                ):
            # 03. Process the results for each fold-removed
            # subset of the expression data for the current
            # comparison
            folds = [f'{_}' for _ in np.arange(1,N_FOLDS_KFCV+1)]
            for fold in folds:
                print(f'Processing fold {fold} results for this comparison...')
                fold_str = f'fold_{fold}'
                results_file = [_ for _ in results_files if fold_str in _][0]
                norm = [_ for _ in norm_count_files if fold_str in _][0]
                res_df = pd.read_csv(f'{dea_results_group_dir}/{results_file}',index_col=0)
                norm_df = pd.read_csv(f'{dea_results_group_dir}/{norm}',index_col=0)

                # 03a. Make sure the indices for the results and normed counts DataFrames match
                res_idx = res_df.index.values.tolist()
                norm_idx = norm_df.index.values.tolist()
                idxs_consistent = (res_idx == norm_idx)
                if idxs_consistent:
                    print(f'Result and normalized count indices match. Proceeding...')
                else:
                    print(f'Result and normalized count indices do not match.')

                # 03b. Compute the mean normalized count value for each gene in the results file
                cols_for_mean = [_ for _ in norm_df.columns.tolist() if 'gene_name' not in _]
                norm_df['norm_count_mean'] = np.mean(norm_df[cols_for_mean].to_numpy(),axis=1)
                norm_mean_df = norm_df['norm_count_mean'].copy()
                res_df = res_df.join(norm_mean_df)

                # 03c. Inspect the number of genes with significant log2 fold changes
                # in this comparison
                res_sig_idx = res_df.loc[res_df['padj'] < alpha].index
                res_not_sig_idx = [_ for _ in res_df.index if _ not in res_sig_idx]
                res_sig = res_df.loc[res_df['padj'] < alpha].copy()
                upreg = res_sig.loc[res_sig['log2FoldChange'] > 0]
                downreg = res_sig.loc[res_sig['log2FoldChange'] < 0]
                print(f'Genes with padj < {alpha}: {len(res_sig)} ({len(res_sig)*100.0/len(res_df):0.002}% high-count genes)')
                print(f'\t Upregulated: {len(upreg)} ({len(upreg)*100.0/len(res_df):0.002}%)')
                print(f'\t Downregulated: {len(downreg)} ({len(downreg)*100.0/len(res_df):0.002}%)')

                # 03d. Plot the log2 fold change as a function of mean normed count
                # and color markers by whether the change in expression is significant
                # at the specified alpha level
                plt.figure()
                plt.plot(res_df.loc[res_not_sig_idx]['norm_count_mean'].values.tolist(),
                        res_df.loc[res_not_sig_idx]['log2FoldChange'].values.tolist(),
                        markersize=2,
                        marker='o',
                        linestyle='None',
                        color='gray')
                # 03e. Plot data for genes with significant lfc in the range (-2,2)
                res_sig_lfc_btw_neg_2_2 = res_sig.loc[
                        list(
                            set(res_sig.loc[res_sig['log2FoldChange'] > -2].index.values).intersection(
                            set(res_sig.loc[res_sig['log2FoldChange'] < 2].index.values))
                            )
                        ].copy()
                plt.plot(res_sig_lfc_btw_neg_2_2['norm_count_mean'].values.tolist(),
                        res_sig_lfc_btw_neg_2_2['log2FoldChange'].values.tolist(),
                        markersize=2,
                        marker='o',
                        linestyle='None',
                        color='red')
                res_lfc_gt_2 = res_sig.loc[res_sig['log2FoldChange'] > 2].copy()
                res_lfc_lt_neg2 = res_sig.loc[res_sig['log2FoldChange'] < -2].copy()
                plt.plot(res_lfc_gt_2['norm_count_mean'].values.tolist(),
                        res_lfc_gt_2['log2FoldChange'].values.tolist(),
                        markersize=2,
                        marker='^',
                        linestyle='None',
                        color='red')
                plt.plot(res_lfc_lt_neg2['norm_count_mean'].values.tolist(),
                        res_lfc_lt_neg2['log2FoldChange'].values.tolist(),
                        markersize=2,
                        marker='v',
                        linestyle='None',
                        color='red')
                plt.xscale('log')
                plt.ylim((-2,2))
                plt.xlabel('mean of normalized counts')
                plt.ylabel('log2 fold change')
                plt.title(tc_curr_str)
                plt.savefig(f'{dea_results_group_out_dir}/lfc_raw_vs_mean_norm_counts__{tc_curr_plt_str}__fold_{fold}.png')
                plt.close()

            else:
                print(f'No results files were found for this group; skipping.\n')


    print(f'\n')


 
