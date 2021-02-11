# GMN-DDC

## Detecting Duplicate Contributions in Forked-based Development via Graph Matching Network

Python library dependencies:
+ tensorflow -v : 1.13.1
+ numpy -v : 1.18.5
+ nltk -v : 3.4.5
+ flask -v : 1.1.1
+ GitHub-Flask -v : 3.2.0
+ gensim -v : 3.8.3
+ scipy -v : 1.4.1 
+ others: sklearn, bs4,

---

Dataset:

[dupPR]: Reference paper: Yu, Yue, et al. "A dataset of duplicate pull-requests in github." Proceedings of the 15th International Conference on Mining Software Repositories. ACM, 2018. (link: http://yuyue.github.io/res/paper/DupPR-msr2017.pdf)
<including: 2323 Duplicate PR pairs in 26 repos>

[dupPR for training set](https://github.com/lyl098yst/DDC-GMN/blob/main/data/clf/first_msr_pairs.txt)

[dupPR for testing set](https://github.com/lyl098yst/DDC-GMN/blob/main/data/clf/second_msr_pairs.txt)

[Non-duplicate PRs for training set](https://github.com/lyl098yst/DDC-GMN/blob/main/data/clf/first_nondup.txt)

[Non-duplicate PRs for testing set](https://github.com/lyl098yst/DDC-GMN/blob/main/data/clf/second_nondup.txt)

---
If you want to use our model quickly, you first need to obtain the data information `getData.py`, then use the obtained information to compose the graph     `getGraph_remove.py`, and use the composed graph to train the `gmn/train.py`, finally use the trained model to test `gmn/getResult.py` 

+ getData.py
  
    `python getData.py GMN 50000`
    
    (It will generate eight .txt files which include all data for getGraph as following,the second parameter indicates that this is a GMN model, and the third parameter indicates the data set size of the negative label used for training )
    `data/clf/first_msr_pairs_pull_info_X.txt`and`data/clf/first_msr_pairs_pull_info_y.txt`
    >Positive label data file for training 
    >
    `data/clf/first_nondup_pull_info_X.txt`and`first_nondup_pull_info_y.txt`
    >Negative label data file for training 
    >
    `data/clf/second_msr_pairs_pull_info_X.txt`and`data/clf/second_msr_pairs_pull_info_y.txt`
    >Positive label data file for testing 
    > 
    `data/clf/test_pull_info_X.txt`and`data/clf/test_pairs_pull_info_y.txt`
    >Negative label data file for testing

    The eight files contains title and description and the similarity of other features                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    
+ getGraph_remove.py
    
    `python getGraph_remove.py 3`
    
    (It will take the .txt files from `getData.py` to generate Graph,The second parameter indicates the size of the sliding window used when composing the graph, and then 3 graph files are generated, as follows )
      
    `remove_XXX_N.train_graphs`、`remove_XXX_N.train_val_graphs`and`remove_XXX_N.test_graphs`
    
    >Note: XXX 可以为` `、`title`、`body`,Respectively indicate that all features are retained during the composition process, the title text is removed, and the body text is removed;
     Where N represents the size of the sliding window used in composition. 

+ gmn/train.py  

    `python gmn/train.py fileName N1 N2`
    
    (It will take `fileName_N1.train_graphs`and`fileName_N1.train_val_graphs` to train the GMN model,and it will generate saved model files.)
    
    >Note: fileName can be `remove`、`remove_title`、`remove_title_body`.The N1 means the slid window size.The N2 means save models after N2 steps. 
    
+ gmn/getResult.py

     `python gmn/getResult.py fileName N1 N2`
    
    (It will take `fileName_N1.test_graphs` to test the GMN model.)
    
    >Note: fileName can be `remove`、`remove_title`、`remove_title_body`.The N1 means the slid window size.The N2 means save models after N2 steps. 
    

---

layers.py: Including the encoder layer, propagator layer, and Aggregator layer of the GMN model

```
class GraphEncoder(snt.AbstractModule)
class GraphPropLayer(snt.AbstractModule)
class GraphAggregator(snt.AbstractModule)
``` 

getData.py: Get data from github using API.

``` python
# Set up the input dataset
# Get the text of title description and the similarity of features 
getData()
```

getGraph_remove.py: Obtain a graph composed of text information and feature similarity.

```
build_graph(start, end) # build graph for train、validate and test
train_graph = build_graph(start=0, end=real_train_size)
train_val_graph = build_graph(start=real_train_size, end=train_size)
test_graph = build_graph(start=train_size, end=train_size + test_size)
``` 

getBatch.py: Combine the data into a batch.

```
batch_graph(dataset, from, batch):# Start from the "from" position of the dataset, take batches and stack them up 
```

train.py: Send the graph composed of the data set to the GMN for training.

```
# Load the data first, then load the initial model
build_placeholders(node_feature_dim, edge_feature_dim)
def build_model(config, node_feature_dim, edge_feature_dim)
def fill_feed_dict(placeholders, batch)
```

getResult.py: Load the model saved during training, use test graph to test the model.
```
# The same as the train.py.But the results used Precesion@k, recall@k, F1@k to evaluate.
```

util.py: Contains the config of the model, but also the calculation formula for similarity and loss.
```
compute_cross_attention(x, y, sim)
euclidean_distance(x, y)
airwise_loss(x, y, labels, loss_type="margin", margin=0.0)
get_default_config()
```

nlp.py: Natural Language Processing model for calculating the text similarity.


``` python
m = Model(texts)
text_sim = query_sim_tfidf(tokens1, tokens2)
``` 


comp.py: Calculate the similarity for feature extraction.

``` 
# Set up the params of compare (different metrics).
# Check for init NLP model.
feature_vector = get_pr_sim_vector(pull1, pull2)
```


---

git.py: About GitHub API setting and fetching.

``` python
get_repo_info('repositories',
              'fork' / 'pull' / 'issue' / 'commit' / 'branch',
              renew_flag)

get_pull(repo, num, renew)
get_pull_commit(pull, renew)
fetch_file_list(pull, renew)
get_another_pull(pull, renew)
check_too_big(pull)
```


fetch_raw_diff.py: Get data from API, parse the raw diff.

``` python
parse_diff(file_name, diff) # parse raw diff
fetch_raw_diff(url) # parse raw diff from GitHub API
```


