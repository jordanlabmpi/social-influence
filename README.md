# social-influence

Code and data repository for:
"Behavioral traits that define social dominance are the same that reduce social influence in a consensus task"
*Rodriguez et al.* 2020

Current preprint:
https://www.biorxiv.org/content/10.1101/845628v2.full

#### Contents

 - **data**
  - mask rcnn annotations and trainings dataset
  - spreadsheeets used for statistical analysis
  - raw tracks and revised (identified and corrected) tracks
  - raw visual field output


- **figures**  
 - all figures of the manuscript  


- **scripts**
  - code for aduino controllers, automatic LEDs and feeders
  - python code for main analysis, network randomization, visual field/raycasting
  - jupyter notebooks for main analysis and visualization
  - R scripts for all models (survival analysis and linear models, additional network randomization test implementation)

In **visualization_tests-networkrandomization.ipynb**, we run all trajectory analyses (individual methods with docstrings in **analysis.py**), generate speeding events and use these to create social aggression networks. Most of the figures are also created in this notebook. Further, network randomization tests and respective visualizations are performed directly by permuting identities early in the analysis.

In **visualization_analysis-visualfield.ipynb**, the visual field reconstructions/ray casting is performed for all trials. The raw output of this is provided in the data directory. We visualize an example for one of the figures. Further, raycasting.py includes a small interactive example of the raycasting.

In **visualization_survival.ipynb**, we visualize the survival model with matplotlib to make it more consistent with the rest of the figures (stats in R, see below).

In **linearmodel.R** we test the effect of social status on noise frequency (i.e. how often individuals swim at elevated speed).

With **networkrandomization.R**, we provide a similar implementation of the network randomization tests (all tests and visualizations were performed with the python code, see above). See the file for differences of the implementations.

In **survivalanalysis.R**, we fit a Cox proportional hazards model to check whether dominant and subordinate individuals have different social influence in the group consensus trials.

For details on Mask-RCNN training, inference and individual tracking, check out our other preprint (accepted at *Movement Ecology*):
https://www.biorxiv.org/content/10.1101/2020.02.25.963926v1.full  
This preprint provides all necessary methods to reproduce the training, inference and tracking for this study with its additional files. We will link an actively maintained repository here once the methods paper is published. We provide the image annotations in the data directory. Videos can be made available on request.