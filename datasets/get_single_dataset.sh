#!/bin/bash
# Download a single dataset folder from the RehabPile archive instead of
# the full ~60-dataset sweep in get_datasets.sh, for use on slow connections.
#
# Usage: ./get_single_dataset.sh <dataset_origin> <folder> <task>
#   e.g. ./get_single_dataset.sh KIMORE_clf_bn Sq classification
#        ./get_single_dataset.sh UIPRMD_clf_bn STS classification

root_path="https://maxime-devanne.com/datasets/RehabPile/"

dataset_origin=$1
folder=$2
target_task=$3

mkdir -p "$target_task/"

new_dataset_name="${dataset_origin}_${folder}"
new_path_download="${target_task}/${new_dataset_name}"

included_dir="/datasets/RehabPile/${dataset_origin}/${folder}"

wget -r -np -nH --cut-dirs=4 -R "index.html*" "$folder" -P "$new_path_download" \
    -I "$included_dir" "${root_path}${dataset_origin}/${folder}"
rm "${new_path_download}/${folder}"

echo "$new_dataset_name"
