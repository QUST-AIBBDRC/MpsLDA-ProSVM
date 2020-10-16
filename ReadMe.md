##MpsLDA-ProSVM

MpsLDA-ProSVM: predicting multi-label protein subcellular localization by wMLDAe dimensionality reduction and ProSVM classifier

###Guiding principles:

**The dataset file contains Gram-negative bacteria dataset, Gram-positive bacteria dataset , plant dataset and virus dataset.

**Feature extraction
1) Evolutionary information: 
  * psepssm.m is the implementation of PsePSSM.
2) Physicochemical_information: 
  * PAAC.m,mainpseaac.m is the implementation of PseAAC.
3) Sequence_information:
  * CTriad.py is the implementation of CT.
4) Annotation information：
  * Gene Ontology can be found from http://www.ebi.ac.uk/GOA/.
** Dimensional reduction:
  * DMLDA_transform.m represents the DMLDA.
  * MDDM_transform.m represents MDDM.
  * PCA_transform.m represents PCA.
  * MLSI_transform represents MLSI.
  * MVMD_transform represents MVMD.

** Classifier:
  * LIFT.m is the implementation of LIFT.
  * MLKNN_test.m,MLKNN_train.m are the implementation of MLKNN.
  *  ML_GKR.m is the implementation of ML_GKR.
  *  ML_RBF_train.m, ML_RBF_test.m is the implementation of ML_RBF.
  *  RankSVM_train.m,RankSVM_test.m is the implementation of RankSVM.
  *  orderSynthetic.m, ProSVM.m is the implementation of ProSVM.

** independent_test:
  *  The independent_test file contains the code of the test of independent dataset.  
  * And you can run the demo.m in MATLAB.

