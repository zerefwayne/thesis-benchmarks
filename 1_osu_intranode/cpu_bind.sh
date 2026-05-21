# cpu_bind.sh — canonical GCD <-> CPU mapping for LUMI standard-g
#
# Lifted from 3_osu_eessi_thesis/osu_allreduce.sh:59-81. The NUMA-correct
# pairing of GCD to nearest CPU core on a LUMI MI250X node:
#
#   GCD 0 -> CPU 49     GCD 1 -> CPU 57   (NUMA 3)
#   GCD 2 -> CPU 17     GCD 3 -> CPU 25   (NUMA 1)
#   GCD 4 -> CPU  1     GCD 5 -> CPU  9   (NUMA 2)
#   GCD 6 -> CPU 33     GCD 7 -> CPU 41   (NUMA 0)
#
# This mapping is documented in the LUMI hardware notes and matches the
# topology dumped by parse_kfd.py + rocm-topo. Do not reorder without
# re-verifying against /etc/cray/xname + rocm-smi --showtoponuma on a node.

cpu_for_rank() {
    local rank=$1
    case $rank in
        0) echo 49 ;; 1) echo 57 ;;
        2) echo 17 ;; 3) echo 25 ;;
        4) echo  1 ;; 5) echo  9 ;;
        6) echo 33 ;; 7) echo 41 ;;
    esac
}

# Comma-separated CPU list for N ranks in order (rank 0 first).
# Used by srun --cpu-bind=map_cpu:$mask and mpirun --cpu-set $mask.
cpu_mask() {
    local n=$1 i=0 mask=""
    while (( i < n )); do
        if (( i == 0 )); then
            mask=$(cpu_for_rank $i)
        else
            mask="${mask},$(cpu_for_rank $i)"
        fi
        ((i++))
    done
    echo "$mask"
}

# Comma-separated CPU list for a pair (g0, g1) — used by pt2pt scripts where
# ROCR_VISIBLE_DEVICES makes rank 0 see g0 and rank 1 see g1.
cpu_for_pair() {
    local g0=$1 g1=$2
    echo "$(cpu_for_rank "$g0"),$(cpu_for_rank "$g1")"
}