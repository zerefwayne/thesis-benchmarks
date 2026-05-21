#!/bin/bash
GCD_A=$1; GCD_B=$2; shift 2
rank=${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-0}}}
case "$rank" in
    0) export ROCR_VISIBLE_DEVICES=$GCD_A ;;
    *) export ROCR_VISIBLE_DEVICES=$GCD_B ;;
esac
exec "$@"
