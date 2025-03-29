#!/bin/bash
#BSUB -P acc_motor # project name
#BSUB -q premium # queue name ('premium' is standard on Minerva)
#BSUB -n 10 # number of tasks in a parallel job (also submits as a parallel job)
#BSUB -R span[hosts=1] # resource requirements
#BSUB -R rusage[mem=20000] # resource requirements
#BSUB -W 10:00 # job runtime limit (HH:MM)
#BSUB -J /sc/arion/projects/motor/WILSOA28/demuxlet_analysis/snrnaseq__05_process_dea_results/log_files/run__05_process_DEA_results__0
#BSUB -o /sc/arion/projects/motor/WILSOA28/demuxlet_analysis/snrnaseq__05_process_dea_results/log_files/run__05_process_DEA_results__0.o
#BSUB -e /sc/arion/projects/motor/WILSOA28/demuxlet_analysis/snrnaseq__05_process_dea_results/log_files/run__05_process_DEA_results__0.e
#BSUB -L /bin/bash

set -ev

# 01. Set up environment
# 01a. Activate conda environment
ml anaconda3/2020.11
ml -python
source /hpc/packages/minerva-centos7/anaconda3/2020.11/etc/profile.d/conda.sh
conda activate CO-deg-analysis-env

# 01b. Get root directory
exec_dir=$( pwd )
cd "${exec_dir}"
an_dea_res_dir="${exec_dir}/scripts"

# 02. Set up config files
# 02a. Specify DEA summary config file path
cfg="${exec_dir}/configs/config_SUD_DEG_analysis.yaml"
echo "${cfg}"
# 02b. Specify DEG subcluster config file path
cfg_dsc="${exec_dir}/configs/config_SUD_DEG_subclustering.yaml"
echo "${cfg_dsc}"
# 02b. Add root directory to config file if not specified
if ! $( grep -q 'root_dir' ${cfg} ); then
	echo "Initializing config file with current directory as root directory"
	echo "root_dir: '${exec_dir}'" >> ${cfg}
else
	echo "DEA results summary config file already contains root directory; proceeding"
fi
if ! $( grep -q 'root_dir' ${cfg_dsc} ); then
	echo "Initializing config file with current directory as root directory"
	echo "root_dir: '${exec_dir}'" >> ${cfg_dsc}
else
	echo "DEG subclustering config file already contains root directory; proceeding"
fi

# 03. Specify cell type(s) for processing DESeq2 results
group="dopaminergic_neuron"

# 04. Run post-DEA analysis of DEGs
# 04a. Plot DEA results per k-fold data subset
echo -e "Plotting results for individual fold results...."
python "${an_dea_res_dir}/05a_plot_deg_results_per_kfold_subset.py" --config-yaml-path ${cfg} --group-to-process "${group}"
# 04b. Plot cross-validated DEA results
echo -e "Plotting results for k-fold cross-validated results...."
python "${an_dea_res_dir}/05b_process_kfcv_deg_results.py" --config-yaml-path ${cfg} --group-to-process "${group}"

# 05. Generate comparisons of DEGs across SUD DEAs
# 05a. Generate summary visualizations
echo -e "Generating visual comparisons of DEGs across SUD DEAs...."
python "${an_dea_res_dir}/05c_compare_DEGs_across_DEAs.py" --config-yaml-path ${cfg} --group-to-process "${group}"

# 06. Run DEG subclustering for an example DEA, for an
# example pair of cell types
# 06a. Specify a pair of cell types
groups_DEG_subclusts="dopaminergic_neuron,microglia"
# 06b. Run code to compute DEG clusters for these cell
# types, for a specified DEA
echo -e "Generating pairwise DEG Pearson correlations for specified DEA/cell type DEGs...."
python "${an_dea_res_dir}/05d_gen_ppt_deg_maps.py" --config-yaml-path "${cfg_dsc}" --group-to-process "${groups_DEG_subclusts}"
# 06c. Get DEG clusters
echo -e "Computing DEG clusters...."
python "${an_dea_res_dir}/05e_perform_corr_based_DEG_clustering.py" --config-yaml-path "${cfg_dsc}"
# 06d. Split DEG clusters into putative subclusters
echo -e "Splitting DEG clusters into subblocks/subclusters...."
python "${an_dea_res_dir}/05f_split_DEG_clusters.py" --config-yaml-path "${cfg_dsc}"
# 06e. Get initial GO terms for DEG subclusters
echo -e "Getting gene ontology terms for DEG subclusters...."
python "${an_dea_res_dir}/05g_get_DEG_module_GS_terms.py" --config-yaml-path "${cfg_dsc}"
# 06f. Pull highly coordinated, mixed-cell-type DEG subclusters,
# and generate skeleton files with organized GO terms to facilitate manual functional annotation
echo -e "Generating skeleton functional annotation documents for highly coordinated, mixed-cell-type DEG subclusters...."
python "${an_dea_res_dir}/05h_process_all_trtmnt_DEG_cassettes.py" --config-yaml-path "${cfg_dsc}"



