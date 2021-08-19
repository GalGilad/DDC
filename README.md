# DDC
A data-driven approach for constructing mutation categories for mutational signature analysis

DDR_genes.npy includes a list of DNA damage repair genes from https://www.mdanderson.org/documents/Labs/Wood-Laboratory/human-dna-repair-genes.html

The two mutation opportunity python dictionaries are of {'7-mer sequence': occurrences} - one for GRCh37 genome and the other for GRCh37 exome.

example_* files are input samples. Full size gene expression and mutation data can be downloaded from ICGC data portal (https://dcc.icgc.org/) [1] and should be processed to meet the format of the example files.

Dependencies: rcca (https://github.com/gallantlab/pyrcca) ; sklearn ; pandas ; numpy ; matplotlib

categorization_learning.py and categorization_evaluation.py scripts are independent of each other.

# categorization_learning.py
How to run:
- python categorization_learning.py --data example --opt_k 5

In this case, the script will load example_catalog.csv , example_ge.csv , example_wgs_catalog.csv files

List of parameters:

(general parameters)
- --data_name: names of datasets separated with ,
- --opt_k: optimal K values separated with ,

(genetic algorithm parameters)
- --m: .01 / .05 / .1; default=.05; mutation rate
- --q: 0 / .5 / / .7 / 1; default=.5; crossover rate
- --p: 3 / 5 / 7; default=7; selection power
- --c: 1 / 5 / 10 / 20; default=10; center of mass of category size dirichlet dist.

# categorization_evaluation.py
How to run:
- python categorization_evaluation.py

Configuration of datasets is based on the config.json file. Current configuration will run the script with the example data, assuming optimal_k=5 and activity of COSMIC signatures 1-4. To add more datasets to the evaluation script, add datasets under "datasets" in the config.json file.

[1] I. C. G. Consortium. International network of cancer genome projects. Nature, 464(7291):993{998, Apr 2010. 20393554[pmid].
