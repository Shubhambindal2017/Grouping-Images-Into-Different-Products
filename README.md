# Grouping The Images Into Different Products
#### Motivation Behind the used approach
I wasn't able to found any dataset for the desired problem on which I can train a full-classifier or a classifier by fine-tuning a pre-trained classifier.
#### Approach Used : Image Clustering Using Transfer Learning
- Image Clustering using by simply flattening the image and passing it to clustering algorithm doesn't preseve image features.
- Instead, Convolutional Neural Networks preserves important characteristics of an image, CNN layers detects pixels, edges, text, parts, objects in the image, thereby preserving all the important features of an image.
- I used pre-tained VGG19 and ResNet50 on imagenet for feature extraction.
- Also used 2 clustering methods [KMeans and AgglomerativeClustering (as it assumed to be better for small dataset)] with each pre-trained model and find silhouette scores with different number of clusters, to know which feauture-extraction model, clustering method, and number of clusters are giving the max avg silhouette score.
- The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample. The Silhouette Coefficient for a sample is (b - a) / max(a, b). To clarify, b is the distance between a sample and the nearest cluster that the sample is not a part of. The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters. 

Transfer Learning
Core idea is instead of building a Convolutional Neural Network from scratch to solve our task, what if we can reuse existing trained models like VGG16, AlexNet architectures.
Keras actually has VGG16 trained on ImageNet dataset, which is the one of the largest object classification dataset.
