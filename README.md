# GAT_proteomics
## proteomics network classification with GAT <h2> 
### Workflow <h3> 
![GitHub Logo](/images/githubpagefig.png)

# Prerequisites
  Users need to install python (https://www.python.org/downloads/) and some python packages:
   * [pytorch]
   * [dgl]
   * [numpy]
   * [pandas]
   * [networkx]
   * [matplotlib]
   
# Data preparation and Model training
 1. Add edge file, node feature file, and label file of a network/a set of networks into folder "Data". <h4> 
 2. For graph classification, a set of graphs are needed.<h4> 
    Run python script “graph_classification.py” to train and validate a GAT model. The well-trained model will be stored in folder "Model". <h4> 
    python graph_classification.py root edge.file node.feature.file graph.label <h4> 
    Run python script "graph_evaluation.py.py" to use the trained model make prediction on other dataset. <h4> 
    python graph_evaluation.py rt model node.feature.file<h4> 
 3. For graph classification, one graphs is needed.
    Run python script “model_training.py” to train and validate a GAT model. The well-trained model will be stored in folder "Model".   <h4>
    python model_training.py root edge.file node.feature.file graph.label <h4> 
    Run python script "model_evaluation.py.py" to use the trained model make prediction on other dataset. <h4> 
    python model_evaluation.py rt model node.feature.file<h4> 
  
