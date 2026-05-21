#!/bin/bash
local_rank=${SLURM_LOCALID:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}
export ROCR_VISIBLE_DEVICES=$local_rank
exec "$@"
