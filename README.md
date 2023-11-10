# Evaluation of Sampling Methods for Discovering Facts from Knowledge Graph Embeddings

# Table of Contents

1. [Abstract](#abstract)
2. [Model Training](#model-training)
3. [Fact Discovery](#fact-discovery)
4. [Reproducing Our Results](#reproducing-our-results)

## Abstract
Knowledge graphs (KGs) are being used in many real-world application domains, ranging from search engines to biomedical data analysis. Even if there is a large corpus of KGs available, they are inherently incomplete due to the incompleteness of the sources based on which they were constructed. Knowledge graph embeddings (KGEs) is a very popular technique to complete KGs. However, they are only capable of answering true or false to a given fact. Thus, users need to provide a concrete query or some test data. Unfortunately, such queries or data are not always available.  There are cases where one wants to discover all (or as many as possible) missing facts from an input KG given a KGE model.  To do so, one should provide to the KGE model candidate facts consisting of the complement of the input KG.  This can be infeasible even for small graphs simply due to the size of the complement graph. In this paper, we define the problem of discovering missing facts from a given KGE model and refer to it as fact discovery. We study sampling methods to get candidate facts and then using KGEs to retrieve the most plausible ones. We extensively evaluate different existing sampling methods and provide guidelines on when each one of them is most suitable. We also discuss the challenges and limitations that we encountered when investigating the different techniques. With these insights, we hope to shed light and attract more researchers to work on this unexplored direction.

## Model Training
Our experiments include 4 datasets, 5 KG embeddings, resulting in a total of 20 different KGE models. We trained our models using the [LibKGE library](https://github.com/uma-pi1/kge#results-and-pretrained-models). We used pre-trained model already provided on the library's github page and trained and optimized the rest on our own setup as per the instructions in the [ICLR2020](https://github.com/uma-pi1/kge-iclr20) github page. The config files required for the hyperparameter optimization can be found in [config_files](config_files/).

## Fact Discovery
We adopted the discovery algorithm from [Ampligraph](https://docs.ampligraph.org/en/1.4.0/generated/ampligraph.discovery.discover_facts.html) and introduced some minor changes to suit our LibKGE-based experimental code. The original code of Ampligraph deals with the fact candidate generation and evaluation sequentially in a single function call. In contrast, our [discover.py](scripts/discover.py) first generates fact candidates from all datasets using all of the discovery strategies. Then we run [run_eval.sh](scripts/run_eval.sh) to evaluate and filter the fact candidates.

## Reproducing Our Results
For reproducing our experiments, we expect to a certain extent familiarity with the LibKGE framework. Our workflow consist of two main steps:

### Fact Discovery
In this step of our workflow, we discover facts from existing datasets.

```sh
chmod +x run_disc.sh
./run_disc.sh
```

The script calls [discover.py](scripts/discover.py) for the datasets as specified in the *.sh* file. The script requires the train-test-validation split of the dataset saved in a *.del* file, which stores the indexified triple list. An example is provided in [/data](./data/). For each dataset, we discover five (amount of strategies) sets of fact candidates stored in *name_of_strategy.del* within the [/data](./data/) folder. The length of discovery time and the amount of discovered facts is stored in [/discovery_logs](./discovery_logs/)

### Evaluation
In this step, we want to evaluate the discovered facts and filter them based on their rankings.

```sh
chmod +x run_eval.sh
./run_eval.sh
```

The script calls [eval.py](scripts/eval.py) which evaluates each triple in *name_of_strategy.del*, and filtering out those ranking below 500. This is done for all strategies in all datasets using LibKGE embedding models (pytorch checkpoints stored in [/models](/models/)). The list of discovered facts and their respective ranks are stored in [/discovered_facts](/discovered_facts/).

## Customizing Our Setup
There are three important modules that of our workflow that is easily customizable: The dataset, strategy, and embedding model. Search the repository for `Customizable` comments to locate parts of the code to customize.

### Datasets
Our fact discovery framework ([discovery.py](/scripts/discover.py)) extracts entities and relations from *.del* files, the standard format used by LibKGE that stores the triples in their index format. This format of storage serves to accelerate the triple scoring within the LibKGE framework. If this is irrelevant, you can also use the more common *.txt* datasets to extract triples and relations from.

### Strategies
A new sampling strategy can be simply added to our array of if-clauses in [utils.py](/scripts/utils.py) to allocate weights to nodes of the graph.

### Evaluation
Our framework implements evaluation ([eval.py](/scripts/eval.py)) in a for-loop and appraising each triple individually using a model saved as a pytorch checkpoint. This can be easily replaced by other 'more neat' evaluation function calls, e.g., that of Ampligraph, whose implementation only requires a single function call, and returns a list of fact candidates and their ranks.