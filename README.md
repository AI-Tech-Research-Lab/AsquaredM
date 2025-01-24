Guidelines to replicate the results.

## FlatDARTS algorithm

The search is performed on CIFAR-10. On NAS-Bench-201, the evaluation on all datasets is shown after the search since accuracies are precomputed. On DARTS, the found genotype must be added to sota/cnn/genotypes.py and then you can run the train by launching scripts/darts-train.sh selecting the architectures and the seeds.

To run the search the following scripts are used.

The evaluated accuracies of NAS-Bench-201 are reported in nasbench201_info.pt (that we provide)

Into the scripts folder
- DARTS: darts-nasbench-search.sh and darts-search.sh (--sam True for FlatDARTS)
- SDARTS: sdarts-nasbench-search.sh and sdarts-search.sh (--sam True for FlatDARTS)
- DARTS-: dartsminus-nasbench-search.sh and dartsminus-search.sh (--sam True for FlatDARTS)
- BETADARTS: betadarts-nasbench-search.sh and betadarts-search.sh (--sam True for FlatDARTS)
Launch them from the main directory

To be compliant with Lambda-DARTS and DARTSPT code, where the code of the original NAS-Bench-201 repo (https://github.com/D-X-Y/AutoDL-Projects) is used, you need to download the NAS-Bench-201-v1_0-e61699.pth and put it into these paths: Lambda-DARTS/NAS-Bench201 and dartspt/data (create the data folder).

Into the Lambda-DARTS/scripts:
- LAMBDADARTS: run_search_NASBENCH and run_search_DARTS.sh (use the flag --sam for FlatDARTS)
Launch it from the main directory

Into the DARTSPT/exp_scripts: 
- NASBENCH: darts-201.sh for search phase and darts-proj-201.sh --resume_epoch 50 --resume_expid search-darts-201-1 for selection (projection) phase
- DARTS: darts-sota.sh for search phase and darts-proj-sota.sh --resume_expid search-darts-sota-s5-2 for selection (projection) phase
Use the --method=darts-sam or darts-proj-sam to use FlatDARTS during the search phase and the projection phase respectively.
Launch them from the exp_scripts directory

## Geometry of the landscapes

Procedures to visualize the geometry are listed in profile_nasbench.py and profile_darts.py
Into the scripts folder
- path-neighbors.sh: Given two NNs and a dataset, it creates the path three and trains all the networks of the path three on DARTS on the dataset. At the end of the process, it creates the line plot of the path three
- train-neighbors.sh: Given a NN and a dataset, finds all the neighbors of radius N and train all the networks stopping the training at a fixed train loss on DARTS on the dataset. At the end of the process, it creates the histogram of the neighbor three

