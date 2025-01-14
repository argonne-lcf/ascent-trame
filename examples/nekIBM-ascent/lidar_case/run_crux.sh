#!/bin/bash
source /home/siramok/crux/ascent-trame/examples/nekIBM-ascent/sourceme
if [ -e "ascent_actions.yaml" ]; then
  rm ascent_actions.yaml
fi
cp terminal.yaml ascent_actions.yaml
NODES=`wc -l < $PBS_NODEFILE`
RANKS_PER_NODE=32
TOTAL_RANKS=$(( $NODES * $RANKS_PER_NODE ))
mpiexec -n $TOTAL_RANKS -ppn $RANKS_PER_NODE ./nek5000

