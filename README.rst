"We will reduce taxes" - Identifying Election Pledges with Language Models
--------------------------------------------------------------------------

Reproducibility files for the paper *"We will reduce taxes" - Identifying Election Pledges with Language Models*

pdf at https://aclanthology.org/2021.findings-acl.301.pdf

logs.zip contains the logs of the experiments described in the paper.

The fold jupyter_xsl_preproc_210130170501 contains the data set, in xlsx format.

For reproducibility:

1. Run preproc.py to create the lookup tables for the experiments (insert the path to the FastText embeddings at the lines 69-70 of step210125.py);
2. In exp210131.py, set the fold created by preproc.py in the -dir_data parameter;
3. In exp210131.py, set the other papameters as you wish;
4. Run exp210131.py!
5. For a nice results' visualization, run printcsv.py -dir exp210131

For any question, please contact me at fornaciari@unibocconi.it