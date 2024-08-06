# Accessing_Cluster
Repo explaining how to access imperial cluster for computing resources simplified

In order to connect to the HPC cluster remotely you must first connect to the imperial VPN described here https://www.imperial.ac.uk/admin-services/ict/self-service/connect-communicate/remote-access/virtual-private-network-vpn/. Next you will need to use an SSH client to connect to the HPC server. From Linux or Mac simply open a terminal and run 'ssh username@login.hpc.imperial.ac.uk'. From windows you will need to use a client such as Putty, openSSH or MobaXterm, this guide shows the process for using MobaXterm. 

Download Mobaxterm from https://mobaxterm.mobatek.net/ and open a session. Select session followed by ssh and enter yourusername@login.hpc.imperial.ac.uk in remote host. ![image](https://github.com/ai4ai-lab/Accessing_Cluster/assets/60004524/8fb352e8-3d49-4374-8189-0f557dcbafd1) . 

Next load your project into moba by simple dragging and dropping on the left window: 

![image](https://github.com/ai4ai-lab/Accessing_Cluster/assets/60004524/984b9f02-cb32-4657-98f8-941385965d9c)


## Submitting a job 
A job is defined by a shell script such as job.sh (shown below) 

![image](https://github.com/ai4ai-lab/Accessing_Cluster/assets/60004524/38d18c4d-5651-465b-89ca-6fc467168476)

This example requires the job.sh to be in the same directory as the file you need to run. for most python projects these two modules should suffice but if your program requires other modules you can see which are available using the comand 'module avail' at the terminal. See https://wiki.imperial.ac.uk/display/HPC/Applications for other available applications. 

To submit the job, first run the command 'sed -i 's/\r$//' job.sh ' in the terminal Followed by 'qsub job.sh' . 

Status of the job can then be checked with the command 'qstat' . 

Two files should have been created in the process jobname.exxxx and jobname.oxxxxx which will contain any possible error messages or outputs from you program (if your output is some data file these will have been created where you specify in your code).

