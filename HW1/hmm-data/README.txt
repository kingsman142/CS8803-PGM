NOTE: All code is developed in Python3, so make sure that is installed with the python3 executable readily available. Other than that, no other dependencies are required. You can run the baseline section of the code by running the following commands in order:

1) Extract hmm_data.zip and navigate to the root directory.
	
2) python3 rare_replace.py gene.train gene.train.rare
	
3) python3 count_freqs.py gene.train.rare > gene.train.rare.counts
	
4) python3 hmm.py baseline gene.train.rare.counts gene.test gene_test.p1.out
	
5) python3 eval_gene_gene_tagger.py gene.key gene_test.p1.out
	
NOTE: This part (trigram HMM section of the code) assumes you have already extracted the hmm_data.zip file and ran through the steps to evaluate the baseline model (above). The trigram HMM can be ran and evaluated by running the following commands:
	
1) python3 hmm.py trigram gene.train.rare.counts gene.test gene_test.p2.out
	
2) python3 eval_gene_tagger.py gene.key gene_test.p2.out