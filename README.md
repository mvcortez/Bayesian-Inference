# Bayesian-Inference
These are accompanying codes for "Hierarchical Bayesian models for inference in biochemical reactions with delays". 

Samplers (Coded sampling Algorithms found in paper):
 1. fixed.py - performs sampling for a hierarchical model with fixed birth delays.
 2. fnorm.py - performs sampling for a model with distributed birth delays. The delay hyperpriors are set as informative folded normal distributions.
 3. mdip.py - performs sampling for a model with distributed birth delays. The delay hyperpriors are set as uninformative Maximum Data Information Prior (MDIP).
 4. nonhier_dist - nonhierarchical counterpart of mdip.py
 5. nonhier_fixed - nonhierarchical counterpart of fixed.py
 6. pooled_nonhier_dist - nonhierarchical counterpart of mdip.py but with pooling of data
 7. pooled_nonhier_fixed - nonhierarchical counterpart of fixed.py but with pooling of data
 8. rational - performs sampling for a model with distributed birth delays. The delay hyperpriors are set as uninformative rational priors.

Data (Raw Data)

Analysis (Performs visualization of samples and gives basic statistics):
 1. quick_analysis_fixed.py - performs basic visualization of samples from posterior obtained using the fixed delay model  
 
