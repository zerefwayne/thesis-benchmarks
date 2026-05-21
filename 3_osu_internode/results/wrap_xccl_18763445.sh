#!/bin/bash
# RCCL/OSU-xccl does its OWN hipSetDevice(local_rank), so each rank must SEE all
# 8 GCDs (unlike the MPI collectives, which use a single ROCR-isolated device).
# Isolating to one GPU here makes every rank's hipSetDevice land on the same
# physical device -> "Duplicate GPU detected" -> RCCL init abort. Expose all.
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
exec "$@"
