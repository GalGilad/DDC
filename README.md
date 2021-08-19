# DDC
A data-driven approach for constructing mutation categories for mutational signature analysis

DDR_genes.npy includes a list of DNA damage repair genes from https://www.mdanderson.org/documents/Labs/Wood-Laboratory/human-dna-repair-genes.html

The two mutation opportunity python dictionaries are of {'7-mer sequence': occurrences} - one for GRCh37 genome and the other for GRCh37 exome.

example_* files are input samples. Full size gene expression and mutation data can be downloaded from ICGC data portal (https://dcc.icgc.org/) [1] and should be processed to meet the format of the example files.

[1] I. C. G. Consortium. International network of cancer genome projects. Nature, 464(7291):993{998, Apr 2010. 20393554[pmid].
