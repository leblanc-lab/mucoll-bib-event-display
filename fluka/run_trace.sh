#!/bin/bash
# Trace one forced muon decay: every FLUKA tracking step goes to trace001_TRACKS.dat.
#
#   run_trace.sh <name> <zlo_cm> <zhi_cm> <seed> [traces_dir]
#   e.g. run_trace.sh z030 -3100 -2900 13
#
# The muon trajectory file is cut to the window zlo < z < zhi, so the decay happens there.
# The position is chosen by hand: these are illustrations, not a sample of the decay rate.
# Needs FLUKA on the PATH (rfluka) and
#   BIB_INPUT   lattice_v_0p8_EU_BIB_.inp
#   BIB_TRAJ    the trajectory file shipped with it (fluka_inputs/run/TRAJb.dat)
#   TRACE_EXE   the executable built with the patched mgdraw (see README.md)
# One decay writes 0.1 to 7 GB.
set -e
NAME=$1; ZLO=$2; ZHI=$3; SEED=$4; DIR=${5:-${BIB_TRACES:-traces}}
: ${BIB_INPUT:?} ${BIB_TRAJ:?} ${TRACE_EXE:?}
mkdir -p $DIR/$NAME && cd $DIR/$NAME
awk -v a=$ZLO -v b=$ZHI '$4 >= a && $4 <= b' $BIB_TRAJ > TRAJb.dat
sed -e "s/^START.*/START     1.0       /" \
    -e "s/^RANDOMIZ.*/RANDOMIZ         1.0$(printf %10.1f $SEED)/" $BIB_INPUT > trace.inp
rfluka -e $TRACE_EXE -N0 -M1 trace.inp > rfluka.log 2>&1
ls -la trace001_TRACKS.dat trace001_DET_IP.dat
