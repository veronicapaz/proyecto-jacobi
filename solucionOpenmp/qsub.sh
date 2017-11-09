#!/bin/sh

# embedded options to qsub - start with #PBS
# -- Name of the job ---
#PBS -N HPC_poisson_problem
# -- specify queue --
#PBS -q hpcintro
# -- estimated wall clock time (execution time): hh:mm:ss --
#PBS -l walltime=01:30:00
# -- number of processors/cores/nodes (request X node with X cores) --
#PBS -l nodes=1:ppn=20
# -- CPU type --
#PBS -l feature='XeonE5-2680'
# -- user email address --
#PBS -M s151457@student.dtu.dk
# -- mail notification when begins(b), ends(e) and fails (a=aborted) --
#PBS -m a

# -- run in the current working (submission) directory --
if test X$PBS_ENVIRONMENT = XPBS_BATCH; then cd $PBS_O_WORKDIR; fi

THREADS=( 2 4 8 9 10 11 12 13 14 15 16 17 18 19 20 )
N=( 1500 )
#METHOD=( jacobi parallel_jacobi parallel2_jacobi gaussseidel )
METHOD=( parallel_jacobi parallel2_jacobi )

# waiting threads should be active and consume CPU power - better for performance
export OMP_WAIT_POLICY=active
# get physical cores numactl should bind threads to
export CORES=`cat /dev/cpuset/torque/$PBS_JOBID/cpus`

for t in ${THREADS[@]} ; do
	for i in ${METHOD[@]} ; do
		for j in ${N[@]} ; do
			outfile="${i}-fast-${CPUTYPE}-Threads_${t}-N_${j}.dat"
			export OMP_NUM_THREADS="$t"
			export SUNW_MP_WARN=true
			/usr/bin/time -o "$outfile" numactl -l -C $CORES ./poisson $j 0 $i >> "$outfile" 2>&1

			NPROCS=`wc -l < "${PBS_NODEFILE}"`
			(
			        echo == CPUTYPE: $CPUTYPE
				echo == CORES: $CORES
				echo == THREADS: $t
			        echo == NPROCS: $NPROCS
			        echo == PBS_NP: $PBS_NP
			        echo == PBS_NODENUM: $PBS_NODENUM
	       			echo == PBS_NUM_NODES: $PBS_NUM_NODES
	        		echo == PBS_NUM_PPN: $PBS_NUM_PPN
	        		echo == PBS_TASKNUM: $PBS_TASKNUM
			) >> "$outfile" 2>&1
		done
	done
done