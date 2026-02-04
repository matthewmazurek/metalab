# Executors

Executors control how experiment runs are dispatched. Choose from thread-based execution for debugging, process-based for local parallelism, or Slurm for HPC clusters.

## Local Executors

::: metalab.ThreadExecutor

::: metalab.ProcessExecutor

## SLURM Executor

For HPC cluster execution via direct `sbatch` submission.

::: metalab.executor.slurm.SlurmExecutor

::: metalab.executor.slurm.SlurmConfig
