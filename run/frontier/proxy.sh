#!/usr/bin/bash

CMD="make jai_forthip3"

## space before and after brackets
#if [ "${SLURM_PROCID}" = "0" ]
#then
#	#rocprof --hip-trace --hsa-trace -i roofline.txt -d $1 ./ocean_model
#	#rocprof --hip-trace -i roofline.txt -d $1 ./ocean_model
#	rocprof --hip-trace --hsa-trace --timestamp on -o results_${SLURM_PROCID}.csv -t $1/hsa_${SLURM_PROCID} ${CMD}
#else
#	${CMD}
#fi

#rocprof --hip-trace --hsa-trace --timestamp on -o rocprof_trace.csv -t data_trace ${CMD}
rocprof -i roofline.txt --timestamp on -o roofline.csv -t data_roofline ${CMD}
#rocprof -i basicperf.txt --timestamp on -o basicperf.csv -t data_basicperf ${CMD}
