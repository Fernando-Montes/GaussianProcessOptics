import sys
import io
import os
import shutil
#import simfiles

theta = [0.001,0.016,0.032,0.048]
eps = [0.02,0.06,0.10,0.14,0.18,0.22]

if(int(sys.argv[1])==0 ):
  usePI = False
else:
  usePI = True
print(sys.argv[1], usePI)

cmds = []
if usePI:
  cmd = "python3 GaussianProcess.py 0 100 "
else:
  cmd = "python3 GaussianProcess.py 0 100 "

runs = []

theta_n = 10

for i in range(len(eps)):
  for j in range(len(theta)):
    if usePI:
      cmds.append(cmd + str(theta[j])+' ' + str(eps[i])+' $SLURM_ARRAY_TASK_ID'+' -P')
    else:
      cmds.append(cmd + str(theta[j])+' ' + str(eps[i])+' $SLURM_ARRAY_TASK_ID')
    if usePI:
      runs.append(str(theta[j])+'_'+str(eps[i])+'_PI')
    else:
      runs.append(str(theta[j])+'_'+str(eps[i]))

alphaL = range(len(cmds))
alphaD = 0.3333333333
restart = True

for a in alphaL:
    runname = runs[a]
    cmd = cmds[a]
    filename = "run_"+runname+".sh"
    fullpath = filename
    with io.FileIO(fullpath, "w") as file:
            file.write("#!/bin/bash -login\n")
            file.write("\n")
            file.write("### define resources needed:\n")
            file.write("### walltime - how long you expect the job to run\n")
            file.write("#SBATCH --time=04:00:00\n")
            file.write("\n")
            file.write("### nodes:ppn - how many nodes & cores per node (ppn) that you require\n")
#            file.write("#SBATCH -n 1 -c 21\n") #,feature=\"lac\"\n")
            file.write("#SBATCH --nodes=-5 --ntasks=20\n") #,feature=\"lac\"\n")
            file.write("\n")
            file.write("### mem: amount of memory that the job will need\n")
            file.write("#SBATCH --mem=30G\n")
            file.write("### you can give your job a name for easier identification\n")
            file.write("#SBATCH -J "+runname + "\n")
            file.write("\n")
            file.write("### error/output file specifications\n")
            file.write("#SBATCH -e /mnt/home/herman67/cosy/iterations/job_outputs/"+runname+"_job_output.txt-%a\n")
            file.write("#SBATCH -o /mnt/home/herman67/cosy/iterations/job_outputs/"+runname+"_job_output.txt-%a\n")
            file.write("### load necessary modules, e.g.\n")
            file.write("#SBATCH --mail-user=herman67@msu.edu\n")
            file.write("#SBATCH --mail-type=FAIL\n")
            file.write("#SBATCH --array=1-100\n")
            file.write("\n")
            file.write("\n")
            file.write("\n")
            file.write("### change to the working directory where your code is located\n")
            file.write("cd /mnt/home/herman67/cosy/iterations/\n")
            file.write("make -j12\n")
            file.write("### call your executable\n")
            file.write(cmd + "\n")
            file.write("scontrol show job ${SLURM_JOB_ID}\n")
