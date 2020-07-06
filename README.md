# Grouping Images Into Different Products
#### Motivation Behind the used approach
I wasn't able to found any dataset for the desired problem on which I can train a full-classifier or a classifier by fine-tuning a pre-trained classifier.
#### Approach Used : Image Clustering Using Transfer Learning
- Image Clustering using by simply flattening the image and passing it to clustering algorithm doesn't preseve image features.
- Instead, Convolutional Neural Networks preserves important characteristics of an image, CNN layers detects pixels, edges, text, parts, objects in the image, thereby preserving all the important features of an image.
- I used both pre-tained VGG19 and ResNet50 on imagenet for feature extraction one by one.
- Also used 2 clustering methods [KMeans and AgglomerativeClustering (as it assumed to be better for small dataset)] with each pre-trained model and find silhouette scores with different number of clusters, to know which feauture-extraction model, clustering method, and number of clusters are giving the max avg silhouette score.
- The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample. The Silhouette Coefficient for a sample is (b - a) / max(a, b). To clarify, b is the distance between a sample and the nearest cluster that the sample is not a part of. The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters. 

```
Maximum Average Sihouette score: [MASs] 
                                                        KMeans Clustering                         AgglomerativeClustering
    Feature Extracted Using            [MASs]              | Clusters associated        [MASs]              | Clusters associated 
          VGG19                        0.19087500870227814 | 6                          0.19320611655712128 | 7
          ResNet50                     0.31846266984939575 | 7                          0.31846266984939575 | 7

```

**So it is clear that the if num of clusters = 7 and trained the model using ResNet50, it gives best max average sihouette score (0.31846266984939575) so far and for both Kmeans and AgglomerativeClustering it is same, so we can use any one of them, To cluster and group images I used AgglomerativeClustering.**

Created 2 py files, utils.py which has functions to extract features from an image given a pre-trained model, and used to get pre-trained models and predictions from a clustering method while sihouette_scores.py plots the average sihouette scores and return the max average sihouette score and its corresponding number of clusters.

Both of these py files are used by Driver notebook, which performs all the experiments explained above, then used the AgglomerativeClustering to make 7 clusters using the features extracted by ResNet50 and then with the help of this make predictions and group images into different products and then stores them accordingly in the Products folder.



