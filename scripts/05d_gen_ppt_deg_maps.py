import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
import scipy.stats as st
from scipy.spatial.distance import pdist,squareform
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import os
import sys
import argparse
import yaml
from utils.plot_heatmap_brackets import *

# This script pulls DEGs for a specified DEA
# and list of cell types, computes DEG pair
# correlations in mean expression across the
# DEA's treatment/condition group donors (using
# the Pearson correlation), and stores these
# values, along with their significances (a beta
# assumption Pearson significance test p-value),
# for DEG cluster and subcluster identification.

# Script setup

# 00. Create argparse object for reading input arguments
parser = argparse.ArgumentParser(description='Visualization of differentially expressed genes' + \
        ' for selected expression data partitions.')
parser.add_argument(f'--config-yaml-path',type=str,required=True)
parser.add_argument(f'--group-to-process',type=str,required=True)
args = parser.parse_args()

# 00a. Get color dictionary
color_dict = mcolors.CSS4_COLORS

# 01. Get analysis parameters from configuration file
cfg = None
with open(args.config_yaml_path) as f:
    cfg = yaml.safe_load(f)
# 01a. Dataset information
# 01a.i. Data root directory
qc_root_dir = cfg.get('root_dir')
# 01a.ii. Input expression data directory
input_data_dir = qc_root_dir + f'/' + cfg.get('input_data_dir')
# 01a.iii. Input expression data file name
input_data_fn = cfg.get('input_data_fn')
# 01a.iv. DEA results input directory
dea_data_dir = qc_root_dir + f'/' + cfg.get('dea_data_dir')
dea_diffexpr_dir = dea_data_dir + f'/' + cfg.get('dea_deseq_formatted_data_dir')
dea_results_dir = dea_diffexpr_dir + f'/' + cfg.get('dea_results_dir')
dea_formatted_dir = dea_diffexpr_dir + f'/' + cfg.get('dea_ready_deseq_formatted_data_dir')
# 01a.v. Metadata file name tag for DEA-ready metadata files
dea_fn_tag = cfg.get('dea_ready_deseq_formatted_data_fn_tag')
# 01a.vi. DEG clustering information output directory
ppt_out_dir = qc_root_dir + f'/' + cfg.get('deg_clust_data_dir')
if not os.path.isdir(ppt_out_dir):
    os.system(f'mkdir -p {ppt_out_dir}')
# 01a.vii. Name tag with cell type mix (for saving cell-type-mix-
# specific results)
cell_type_mix_tag = cfg.get('cell_type_mix_tag')

# 01b. DEA parameters for DEG analysis and visualization
# 01b.i. Upregulated DEG file name tag
upreg_deg_fn_tag = cfg.get('dea_output_upreg_fn_root')
# 01b.ii. Downregulated DEG file name tag
downreg_deg_fn_tag = cfg.get('dea_output_downreg_fn_root')
# 01b.iii. DEA partition variable
grouping = cfg.get('dea_partition_variable')
# 01b.iv. DEA list
test_comparisons = cfg.get('dea_cond_comparison_list')
# 01b.v. Number of folds used in DEA (for pulling metadata files)
N_FOLDS_DEA = cfg.get('dea_n_folds_kfcv')
# 01b.vi. Significance threshold alpha for Pearson correlations
alpha = cfg.get('deg_pearson_corr_sig_alpha')
# 01b.v. RNG random seed to use
rand_seed = cfg.get('rand_seed_deg_corr')
# 01b.v.A. Set the numpy RNG random seed
np.random.seed(seed=rand_seed)
# 01b.vi. Flag indicating whether to analyze control group donors
# instead of the default treatment group donors (as a negative control)
to_do_controls_only = cfg.get('analyze_control_group_flag')
# 01b.vii. Cell type abbreviation map
ct_abbr_map = cfg.get('ct_to_abbreviation_dict')

# 01c. Get list of cell types to analyze
grouping_values = args.group_to_process
if grouping_values == 'None':
    grouping_values = [
            _ for _ in os.listdir(dea_results_dir)
            if os.path.isdir(f'{dea_results_dir}/{_}')
            ]
else:
    grouping_values = [_.strip() for _ in grouping_values.split(',')]
gv_str = '\n'.join(grouping_values)
print(f'Grouping values to be processed:\n{gv_str}')

# 01d. Set gene name prefix from expression data
gene_name_prefix = 'GRCh38_______________'

# 02. Import expression data (QCed, preprocessed, cell typed,
# with countlike expression values)
ahc_fn = f'{input_data_dir}/{input_data_fn}'
adatas_human_qc = ad.read_h5ad(ahc_fn)
print(adatas_human_qc,'\n')
print(list(np.unique(adatas_human_qc.obs['patient_ID'].values.tolist())))

# 02a. Normalize expression per nucleus (to 
# reduce influence on analysis of technical
# sources of internuclear variability)
if 'scaledX' in adatas_human_qc.obsm.keys():
    print(f'Removing existing scaledX object...')
    del adatas_human_qc.obsm['scaledX']
sc.pp.normalize_total(adatas_human_qc)

# 03. For each DEA ('test condition comparison'),
# build a cell-type-blocked DEG x donor matrix
# with mean DEG expression
for tc_curr in test_comparisons:
    tc_str = f'{tc_curr[0]}: {tc_curr[1]} vs. {tc_curr[2]}'
    print(f'Pulling DEGs by cell type for\n\t{tc_str}')
    tc_out_str = '_'.join(tc_curr)
    out_dir_curr = f'{ppt_out_dir}/{tc_out_str}/{cell_type_mix_tag}'
    if not os.path.isdir(out_dir_curr):
        os.system(f'mkdir -p {out_dir_curr}')

    # 03a. Initialize lists for treatment and
    # control group donor IDs
    pts_treatment = []
    pts_control = []

    # 03b. Get DEG files for each specified cell type
    tc_curr_fn_tag = '_'.join(tc_curr)
    ct_deg_dict = {}
    for g in grouping_values:
        print(f'Processing results for {grouping} group {g}...')
        dea_results_group_dir = f'{dea_results_dir}/{g}'
        files = os.listdir(dea_results_group_dir)
        up_deg_files = [
                _ for _ in files if 
                all([ tag in _ for tag in
                    [tc_curr_fn_tag,upreg_deg_fn_tag]
                    ])
                ]
        down_deg_files = [
                _ for _ in files if 
                all([ tag in _ for tag in
                    [tc_curr_fn_tag,downreg_deg_fn_tag]
                    ])
                ]
        udf_str = '\n\t\t'.join(up_deg_files)
        ddf_str = '\n\t\t'.join(down_deg_files)
        print(f'\tUp DEG files:\n\t\t{udf_str}')
        print(f'\n\tDown DEG files:\n\t\t{ddf_str}')

        # 03d. Store up- and down- DEGs for the current cell type
        up_degs = []
        down_degs = []
        if len(up_deg_files) > 0:
            print(f'\tReading up DEGs...')
            up_deg_fn = up_deg_files[0]
            up_deg_full_fn = dea_results_group_dir + f'/{up_deg_fn}'
            up_deg_df = pd.read_csv(
                    up_deg_full_fn,index_col=0
                    )
            print(f'\t{len(up_deg_df)} up DEGs found')
            up_degs = up_deg_df.index.values.tolist()
        if len(down_deg_files) > 0:
            print(f'\tReading down DEGs...')
            down_deg_fn = down_deg_files[0]
            down_deg_full_fn = dea_results_group_dir + f'/{down_deg_fn}'
            down_deg_df = pd.read_csv(
                    down_deg_full_fn,index_col=0
                    )
            print(f'\t{len(down_deg_df)} down DEGs found')
            down_degs = down_deg_df.index.values.tolist()
        ct_deg_dict[g] = {'up': up_degs,'down': down_degs}
        
        # 04. Get donor metadata for the current DEA and cell type
        # to pull donor IDs (per cell type because the donors with nuclei
        # can change for different cell types)
        metadata_dir_curr = dea_formatted_dir + f'/{g}' 
        metadata_fn_tags = [f'{g}{dea_fn_tag}__{_}' for _ in range(1,N_FOLDS_DEA+1)]
        metadata_fn_list = [
                _ for _ in os.listdir(metadata_dir_curr)
                if all([
                    'metadata' in _,
                    any([tag in _ for tag in metadata_fn_tags])
                    ])
                ]
        test_cond_col = tc_curr[0]
        treatment_level = tc_curr[1]
        control_level = tc_curr[2]
        # 04a. Get treatment and control group donor ID lists
        for mfn in metadata_fn_list:
            m_full_fn_curr = metadata_dir_curr + f'/' + mfn
            m_df = pd.read_csv(m_full_fn_curr,index_col=0)
            pts_treatment.extend(m_df.loc[
                m_df[test_cond_col] == treatment_level
                ]['patient_ID'].values.tolist())
            pts_control.extend(m_df.loc[
                m_df[test_cond_col] == control_level
                ]['patient_ID'].values.tolist())
            
    # 05. Make treatment and control group donor ID lists unique
    pts_treatment = list(np.unique(pts_treatment))
    pts_control = list(np.unique(pts_control))
    # 05a. Pull treatment or control donors for analysis
    pts_to_analyze = []
    pts_to_analyze_dir_name = ''
    if to_do_controls_only == True:
        pts_to_analyze = pts_control
        pts_to_analyze_dir_name = 'control_group'
    else:
        pts_to_analyze = pts_treatment
        pts_to_analyze_dir_name = 'treatment_group'
    # 05b. Specify results output directory
    out_dir_curr += f'/{pts_to_analyze_dir_name}'
    if not os.path.isdir(out_dir_curr):
        os.system(f'mkdir -p {out_dir_curr}')
    
    # 06. Compute mean DEG expression values for each donor
    pta_str = ' '.join(pts_to_analyze_dir_name.split('_'))
    print(f'Processing {pta_str}')
    pt_t_df = pd.DataFrame()
    for pt_t_raw in pts_to_analyze:
        pt_t = f'{pt_t_raw}'.rjust(6,'0')
        print(f'\n\tProcessing patient {pt_t}')
        ahc_pt = adatas_human_qc[
                adatas_human_qc.obs['patient_ID'] == pt_t,
                :
                ].copy()
        all_var_names_pt = ahc_pt.var_names.values.tolist()
        
        # 06a. For each cell type, get DEG mean expression levels
        ct_labels = []
        for ct_curr,ud_deg_dict in ct_deg_dict.items():
            degs_reform = [
                    gene_name_prefix+_ for _ in
                    ud_deg_dict['up'] + ud_deg_dict['down']
                    ]
            degs_in_av = [_ for _ in degs_reform if _ in all_var_names_pt]
            ahc_pt_ct = None
            if len(degs_in_av) > 0:
                ahc_pt_ct = ahc_pt[
                        ahc_pt.obs[grouping] == ct_curr,
                        degs_in_av
                        ].copy()
                print(f'Expression data for patient {g} barcodes:',ahc_pt_ct)
                ahc_pt_ct_gene_names = []
                for gn in ahc_pt_ct.var_names.values.tolist():
                    gn_reform = gn.split('GRCh38_______________')[-1]
                    full_gn_reform = ''
                    if gn_reform in ud_deg_dict['up']:
                        full_gn_reform = ct_abbr_map[ct_curr] + '_up_' + gn_reform
                    elif gn_reform in ud_deg_dict['down']:
                        full_gn_reform = ct_abbr_map[ct_curr] + '_down_' + gn_reform
                    else:
                        print(f'Error: cannot find gene {gene_reform} in DEG list.')
                    ahc_pt_ct_gene_names.append(full_gn_reform)
                        
                mean_exp_pt_ct = list(
                        np.mean(np.asarray(ahc_pt_ct.X.todense().copy()),axis=0)
                        )
                ct_labels.extend([ct_abbr_map[ct_curr]]*len(mean_exp_pt_ct))
                for gene_name,mean_exp_curr in zip(ahc_pt_ct_gene_names,mean_exp_pt_ct):
                    pt_t_df.at[gene_name,pt_t] = mean_exp_curr

    # 07. Build DEG pairwise similarity matrix
    # 07a. Remove donors without nuclei for >=1 specified cell type, since
    # they cannot contribute to Pearson correlations for all mixed-cell-type DEG pairs
    # Note: these donors can be identified by nan DEG values
    pt_t_df.dropna(axis='columns',inplace=True)
    # 07a.i. Check for any nans post dropna
    pt_t_df_nans = pt_t_df.loc[pt_t_df.isna().any(axis=1)].index.values.tolist()
    pt_t_df_infs = pt_t_df.loc[pt_t_df.isin([np.inf,-np.inf]).any(axis=1)].index.values.tolist()
    if len(pt_t_df_nans) > 0:
        pt_nan_str = '\n\t'.join(pt_t_df_nans)
        print(f'Warning: the following patients have nan expression values prior to standardization:\n\t{pt_nan_str}')
        print(f'Patient values:')
        for pt_nan_curr in pt_t_df_nans:
            print(pt_t_df.loc[pt_nan_curr].values.tolist())
    if len(pt_t_df_infs) > 0:
        pt_inf_str = '\n\t'.join(pt_t_df_infs)
        print(f'Warning: the following patients have inf expression values prior to standardization:\n\t{pt_inf_str}')
        for pt_inf_curr in pt_t_df_infs:
            print(pt_t_df.loc[pt_inf_curr].values.tolist())
    del pt_t_df_nans,pt_t_df_infs

    # 07b. Standardize DEG mean expression values across remaining donors
    # for future visualization; this step is not necessary for pairwise
    # DEG correlation computations
    pt_t_df_pre_standardization = pt_t_df.copy()
    pt_t_mean = pt_t_df.mean(axis=1)
    pt_t_std = pt_t_df.std(axis=1)
    # 07b.i. Set up handling of standardization for any DEGs with all zero
    # mean counts (meaning all zero counts across all donors; this could occur
    # in control groups or in downregulated DEGs in treatment groups), to ensure
    # that after standardization all values are still zeros.
    means_and_stds = [[m,s] for m,s in zip(pt_t_mean,pt_t_std)]
    # 07b.i.A. Identify DEGs with all zero values 
    all_zero_idxs = [
            i for i,_
            in enumerate(means_and_stds)
            if all([param==0 for param in _])
            ]
    # 07b.i.B. Set the standard deviation for these DEGs
    # artificially to zero, specifically so that standardization
    # results in a zero value, through the transform 0 -> (0-0)/1 = 0
    pt_t_std_idxs = pt_t_std.index.values.tolist()
    all_zero_std_idxs = [
            pt_t_std_idxs[_]
            for _ in all_zero_idxs
            ]
    pt_t_std.loc[all_zero_std_idxs] = 1.0
    # 07b.ii. Compute standardized DEG expression values
    pt_t_df = pt_t_df.sub(pt_t_mean,axis=0)
    pt_t_df = pt_t_df.div(pt_t_std,axis=0)
    # 07b.iii. Check for any nans or infs post-standardization
    pt_t_df_nans = pt_t_df.loc[pt_t_df.isna().any(axis=1)].index.values.tolist()
    pt_t_df_infs = pt_t_df.loc[pt_t_df.isin([np.inf,-np.inf]).any(axis=1)].index.values.tolist()
    if len(pt_t_df_nans) > 0:
        pt_nan_str = '\n\t'.join(pt_t_df_nans)
        print(f'Warning: the following patients have nan expression values post-standardization:\n\t{pt_nan_str}')
        print(f'Patient values:')
        for pt_nan_curr in pt_t_df_nans:
            print(f'Current:',pt_t_df.loc[pt_nan_curr].values.tolist())
            print(f'Prior to standardization:',pt_t_df_pre_standardization.loc[pt_nan_curr].values.tolist())
            print(f'Mean used in standardization:',pt_t_mean.loc[pt_nan_curr])
            print(f'Standard deviation used in standardization:',pt_t_std.loc[pt_nan_curr])
    if len(pt_t_df_infs) > 0:
        pt_inf_str = '\n\t'.join(pt_t_df_infs)
        print(f'Warning: the following patients have inf expression values post-standardization:\n\t{pt_inf_str}')
        for pt_inf_curr in pt_t_df_infs:
            print(f'Current:',pt_t_df.loc[pt_inf_curr].values.tolist())
            print(f'Prior to standardization:',pt_t_df_pre_standardization.loc[pt_inf_curr].values.tolist())
            print(f'Mean used in standardization:',pt_t_mean.loc[pt_inf_curr])
            print(f'Standard deviation used in standardization:',pt_t_std.loc[pt_inf_curr])

    # 07d. Compute DEG pairwise correlations
    # 07d.i. Transpose the DataFrame in preparation for .corr() method
    # (where correlations are computed for each column)
    pt_t_df = pt_t_df.T
    sim_curr = pt_t_df.corr(method='pearson')
    vmin_curr = -1.0
    vmax_curr = 1.0

    # 07e. Plot DEG correlation matrix as a heatmap
    fig,ax = plt.subplots()
    im = ax.imshow(
            sim_curr,
            cmap='inferno',
            vmin=vmin_curr,
            vmax=vmax_curr
            )
    ax.set_xticks(
            np.arange(len(sim_curr)),
            ['']*len(sim_curr)
            )
    ax.set_yticks(
            np.arange(len(sim_curr)),
            ['']*len(sim_curr)
            )
    ct_labels_u = list(np.unique(ct_labels))
    # 07e.i. Draw brackets labeling each cell type's DEGs
    for ctl in ct_labels_u:
        ctlocs = [i for i,q in enumerate(ct_labels) if q==ctl]
        p1 = np.min(ctlocs)
        p2 = np.max(ctlocs)
        print(ax,p1,p2,ctl)
        # 07e.ii. Horizontal brackets
        hb,ht = make_brackets(
                ax,
                is_horiz=True,
                pos1=p1,
                pos2=p2,
                label=ctl
                )
        # 07e.iii. Vertical brackets
        vb,vt = make_brackets(
                ax,
                is_horiz=False,
                pos1=p1,
                pos2=p2,
                label=ctl
                )
    # 07f. Color bar
    cbar = ax.figure.colorbar(im,ax=ax)
    cbar_label = f'Pearson correlation'
    cbar.ax.set_ylabel(
            cbar_label,
            rotation=-90,
            va='bottom'
            )
    pt_heatmap_fn = f'{out_dir_curr}/{test_cond_col}_pts_{treatment_level}_DEG_heatmap.png'
    plt.savefig(
            pt_heatmap_fn,
            dpi=300,
            bbox_inches='tight'
            )
    plt.close('all')

    # 08. Find DEG pairs with significant correlations
    pearson_corr_df = pd.DataFrame(
            columns=[
                'deg_1',
                'deg_2',
                'pearsons_corr_w_beta_test',
                'beta_assumption_pval_two_sided'
                ]
            )
    pc_idx = 0
    all_degs = pt_t_df.columns.values.tolist()
    pc_mtx = np.identity(len(all_degs)).astype(bool)
    for d1_idx,deg_1 in enumerate(all_degs[:-1]):
        for d2_idx,deg_2 in enumerate(all_degs[d1_idx+1:]):
            # 08a. Get the Pearson correlation and built-in
            # significance test p-value. This test uses a beta
            # distribution assumption, which is met if each
            # comparator sample is drawn from a normal distribution
            # of independent samples (which should be roughly
            # satisfied in this scenario, given that each
            # DEG measurement is from a different individual)
            prs_r,prs_p = st.pearsonr(
                    pt_t_df[deg_1].values.tolist(),
                    pt_t_df[deg_2].values.tolist(),
                    alternative='two-sided'
                    )
            pearson_corr_df.at[pc_idx,'deg_1'] = deg_1
            pearson_corr_df.at[pc_idx,'deg_2'] = deg_2
            pearson_corr_df.at[pc_idx,'pearsons_corr_w_beta_test'] = prs_r
            pearson_corr_df.at[pc_idx,'beta_assumption_pval_two_sided'] = prs_p
            # 08a.i. Add a column related to significance based on the specified alpha
            pearson_corr_df.at[pc_idx,f'sig_alpha_beta_assumption_{alpha}'] = (1 if prs_p < alpha else 0)
            # 08b. Add significantly correlated values to a second matrix
            # for visualization
            if prs_p < alpha:
                pc_mtx[d1_idx,d2_idx] = True
                pc_mtx[d2_idx,d1_idx] = True
            pc_idx += 1

    # 09. Save results
    # 09a. Save DEG pairwise correlation and significance results
    pw_deg_corr_sig_fn = f'{out_dir_curr}/{test_cond_col}_pts_{treatment_level}_pw_deg_corr_w_sig.csv'
    pearson_corr_df.to_csv(
            pw_deg_corr_sig_fn,
            index=True
            )
    # 09b. Save the significant pairwise correlation matrix results
    pw_deg_corr_sig_mtx_fn = f'{out_dir_curr}/{test_cond_col}_pts_{treatment_level}_pw_deg_corr_w_sig_mtx.npy'
    np.save(
            pw_deg_corr_sig_mtx_fn,
            pc_mtx
            )

sys.exit()        



