# EcgenDDIM
A DDIM model for producing 2s ECGs

# How to use
Call the main.py with the desired hyperparameters. Any physionet database should do, provided they are cut into 2s intervals. For reproducibility, I can offer the synthetic database I used, just shoot me a message.

# Architectures and Outputs

## BaseDDIM
![BaseDDIM architecture](https://github.com/jackilion/EcgenDDIM/blob/master/img/BaseDDIM_architecture.png)
![BaseDDIM outputs](https://github.com/jackilion/EcgenDDIM/blob/master/img/BaseDDIM_results.png)

## RpeakGuidedDDIM
![RpeakGuidedDDIM architecture](https://github.com/jackilion/EcgenDDIM/blob/master/img/RpeakGuidedDDIM_architecture.png)
![RpeakGuidedDDIM outputs](https://github.com/jackilion/EcgenDDIM/blob/master/img/RpeakGuidedDDIM_results.png)
