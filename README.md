# Fine-grained subphenotypes in acute kidney injury populations based on deep clustering: derivation and interpretation

## Workflow

![Graphic Abstract](https://github.com/YongsenTan/Deep-Consensus-Clustering/blob/main/img/Graphic%20Abstract.png)

- **Study Design:** Individuals with AKI within 48-hour admission were included and temporal laboratory measurements were extracted from two public databases. 
- **Representation Learning:** Laboratory data were split to derive multiple representations with supervised LSTM autoencoders. 
- **Subphenotypes Derivation:** We got the corresponding cluster from multiple representations using K-means. Then, the similarity matrix was derived by computing the frequency of clustering to the same group. Finally, subphenotypes were derived from the similarity matrix by K-means. 
- **Subphenotype Interpretation:** Static patterns analysis, dynamic patterns analysis, and predictive interpretation were conducted to interpret the subphenotypes.



## Understanding deep consensus clustering

![Model Overview](https://github.com/YongsenTan/Deep-Consensus-Clustering/blob/main/img/Model%20Overview.png)

- **a.** Time series electronic health records (EHR) were utilized as input data for the Bi-LSTM encoder in our study. The Bi-LSTM encoder processed the EHR data, and the hidden state in the final time step was extracted as the representation. Subsequently, the Bi-LSTM decoder employed both the representation and the time-reversed input data to reconstruct the original input data. Simultaneously, the supervisor component was co-trained with the Bi-LSTM autoencoder. This supervisor component utilized the extracted representation as input to predict the probability of mortality. 
- **b.** We represented EHR in multiple representations in different hidden dimensions. Subsequently, K-means was employed to cluster each representation. In this illustration, we conducted K-means on five representations (left). The similarity of pair-wise patients was quantified as the frequency of being clustered into the same group. For instance, patient 1 and patient 2 were consistently clustered together in four out of five representations, and their similarity was calculated as 4 divided by 5. This similarity matrix, a symmetrical matrix, was constructed to record the pairwise similarities between patients (right). 

## How to explore this project

### Installing dependencies

All the required dependencies are in [requirements.txt](https://github.com/YongsenTan/Deep-Consensus-Clustering/blob/main/requirements.txt)

For installing all the dependencies run this line of code in command

`pip install -r requirements.txt`

### Run codes

#### Get representations

To generate representations in multiple dimensions in parallel, run code in command

`./01_get_representations.sh`

The parameters were illustrated as bellow,

|   Parameter    | Explanation                                                  |
| :------------: | ------------------------------------------------------------ |
|    max_dim     | max hidden dimensions of LSTM encoder/decoder                |
|  hidden_dims   | hidden dimensions of LSTM encoder/decoder                    |
|   input_dim    | number of input features                                     |
|    n_layers    | number of LSTM layers in encoder/decoder                     |
|   pre_epoch    | number of pre-training epoch of autoencoder without supervisor |
|   lambda_AE    | weight of autoencoder loss                                   |
| lambda_outcome | weight of supervisor loss                                    |

The code results in multiple representations `rep_{n}.pkl` in the folder `./representations` and logs in the folder `./runs`. The representations in different hidden dimensions can be visualized as below

![Visualization of representations](https://github.com/YongsenTan/Deep-Consensus-Clustering/blob/main/img/Visualization%20of%20Representations.png)

#### Conduct consensus clustering

To conduct consensus clustering on representations in multiple dimensions in parallel, run 

[02_consensus_clustering.py](https://github.com/YongsenTan/Deep-Consensus-Clustering/blob/main/02_consensus_clustering.py), which will generates the figure of the relative change in area under CDF Curve and the average consensus value of each cluster. The derived subphenotypes `consensus_cluster_{k}.pkl` will be generated in the folder `./results`. The consensus matrixes demonstrated the similarity of the pair-wised patients. The blue lines indicated the cluster division.

![Graphic Abstract](https://github.com/YongsenTan/Deep-Consensus-Clustering/blob/main/img/Consensus%20Matrixes.png)

#### Conduct further analysis



Further analysis was summarized as below

| Code                          | Description                                                  |
| ----------------------------- | ------------------------------------------------------------ |
| 03_sensitivity_analyze.py     | Sensitivity analysis of consensus matrix by bootstrapping    |
| 04_visualization_of_cm&rep.py | Visualizations of  consensus matrixes and representations    |
| 05_cluster_transfer.R         | Sankey diagram of the transfer of clusters                   |
| 06_relative_risk.R            | Forest plot of relative risk of mortality                    |
| 07_comorbidity_bubble.py      | Bubble plot of the distribution of comorbidities             |
| 08_KDIGO_circlize.R           | Chord diagram of the relationship between subphenotypes and KDIGO stages |
| 09_survival_analysis.R        | Survival analysis and visualization of subphenotypes         |
| 10_KDIGO_dynamic.R            | Bar chart on the dynamic proportion of KDIGO stage           |

