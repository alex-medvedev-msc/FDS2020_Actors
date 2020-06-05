# Replication and extension of "Box office deconfounder"

Here we replicated project "Box office deconfounder" from https://github.com/niksnikson/causalML/tree/master/projects/deconfounder%20latent%20variable%20models/box%20office%20deconfounder . 
We extended it with actor, movie analysis, added three counterfactual queries. 

Queries are:
1) Who can replace Ben Affleck in Dawn of Justice? (Christian Bale)
2) Are there any underrated actors who can play in most expensive movie roles of all time even better than Tom Cruise in Mission: Impossible, Robert Downey Jr. in Avengers, etc. (No)
3) Does Lois Guzman improve every movie? (Yes)

Movie hypotheses:
1) Are non-english movies underrated by simple linear model with only actors as variables? (probably true)
2) Are unpopular (not action or comedy or drama) movie genres also underrated? (probably true)
3) Sequels should be overrated (probably true)

Actor hypotheses:
1) The most underrated actors are mostly from movies with small budgets and they are not well-paid (inconclusive)
2) Cast effect is overrated (true)

**How to use it**

1) Create a conda environment with env.yml file
2) Launch jupyter notebook from it
3) Most of the report are in report.ipynb
4) The assignment model prediction check is in assignment_model_check.ipynb

Code of deconfounder and generative model was adapted from "Box office deconfounder" and refactored into more OOP style. Sadly, pyro is NOT OOP at all, but we did our best. 

The linear regression models, actor analysis, movie analysis and three counterfactuals are in report.ipynb.