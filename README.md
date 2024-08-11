<h1>Song Recommendation Engine</h1>

<h2>Description</h2>
As part of my fulfillment for the UofT SCS Machine Learning course, I have worked on a project building a Spotify Song Recommendation Engine using Machine Learning. The data is procured from Spotify, and it has over 600k tracks between 1921 and 2020; each with 19 characteristics which include valence (musical positiveness that a track exudes), duration (this could be useful as a predictor, if for example, the listener exhibits patterns of only listening to 3-minute songs, this is something we can dig into deeper), artist(s), danceability, popularity, release date, and many more. This is a good, comprehensive dataset as there are many features which is beneficial for our machine learning training
model.

<br />

<h2>Languages and Utilities Used</h2>

- <b>Pandas</b> 
- <b>Spotipy API</b>
- <b>Matplotlib</b>
- <b>seaborn</b>
- <b>scikit-learn</b>

<h2>Data Source</h2>

The Spotify tracks data is procured from Spotify and extracted via API. It was posted as part of a [Kaggle project](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks?select=artists.csv) and is composed of two datasets - Tracks.csv and Artist.csv, used to train and develop our model. Tracks.csv contains 587k unique values and has 20 song features composed of different data types (i.e. numerical, ordinal and categorical).

<p align="center">
Data Distribution: <br/>
<img src="https://i.imgur.com/Sp3CVR1.png" height="80%" width="80%" alt="Data Distribution"/>
<br />
</p>


<h2>Methodology</h2>

The Spotify data set does not contain a target value appropriate for the project objective, i.e. there is no prediction target. Hence, the songs must first be clustered (X variable) into groups and the resultant group number will be its target value (Y value). This is done using an unsupervised technique, K-Means Clustering.
<br />

<p align="center">
Pipelines and Workflow: <br/>
<img src="https://i.imgur.com/1Mvu9hr.png" height="80%" width="80%" alt="Pipelines and Workflow"/>
<br />
</p>

Once X and Y are built, we now have a classification problem, i.e. the data set contains feature variables, X and target variables, Y. Given a new song, classify it into the one of the cluster Y based on the similarities of the songs' features. This can be done by using a myriad of supervised learning models. The following methods will be used to train, finetune and ultimately choosing the best classification model:
- K-Nearest Neighbours
- Decision Tree
- Random Forest

<h2>Unsupervised Learning Model using K-Means Clustering</h2>

K-Means clustering is a popular algorithm used for grouping observations into clusters based on their proximity to centroids. The algorithm aims to identify underlying patterns and behaviors within the data by iteratively assigning each observation to the nearest centroid and then updating the centroids based on the newly formed clusters. The term "means" in K-Means refers to the process of computing the average or centroid of the data points within each cluster. By iteratively optimizing the assignment of observations to clusters, K-Means clustering helps reveal patterns and groupings that exist in the dataset. 

In this clustering approach, numerical attributes were specifically selected as the hyper parameters. These attributes were chosen for their ease of quantification and their suitability for prediction purposes.

<p align="center">
K-Means Label: <br/>
<img src="https://i.imgur.com/xkcE4He.png" height="40%" width="40%" alt="K-Means Label"/>
<br />
</p>

<h2>Supervised Learning Models</h2>

<h3>K-Nearest Neighbours</h3>

In k-NN classification, the output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor. In this project, a pipeline is built using the following elements:
- Standardization: normalizing all numerical parameters to ensure learning biases is
reduced.
- Principal Component Analysis (PCA): transform the feature space of variable X from its
original dimensions to a reduced, three-dimensional representation.

<p align="center">
KNN Pipeline: <br/>
<img src="https://i.imgur.com/xUmgXLY.png" height="30%" width="60%" alt="KNN Pipeline"/>
<br />
</p>

<h3>Decision Tree</h3>

Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation. 

Since DTs requires little data preparation, standardization/normalization of variables is not necessary. Therefore, our transformation pipeline would only include fitting the model for fine-tuning.

<p align="center">
Decision Tree with best hypermeter: <br/>
<img src="https://i.imgur.com/9Pxwdsc.png" height="10%" width="60%" alt="Decision Tree with best hypermeter"/>
<br />
</p>

<h3>Random Forest</h3>

A random forest is an ensemble of decision trees, generally trained via the bagging method (i.e. when sampling is performed with replacement and using the same algorithm for every predictor on different random subset of training set) and usually has all the hyperparameters of a decision tree classifier and bagging classifier to control the ensemble itself. 

The main parameters to adjust when using these methods is number of estimators and number of maximum leaf nodes. The former is the number of trees in the forest. The larger the better, but also the longer it will take to compute. The latter is to control the tree size. Trees will be grown using best-first search where nodes with the highest improvement in impurity will be expanded first. As this is an expansion on decision tree, the pipeline shall be consisting only of fitting the model for fine-tuning.

<p align="center">
Random Forest with best hyperparameter: <br/>
<img src="https://i.imgur.com/rGiypvh.png" height="40%" width="60%" alt="Random Forest with best hyperparameter"/>
<br />
</p>

<h2>Conclusion</h2>

<b>Random Forest classifier </b>appears to have the best result overall however it does not necessarily have a high enough score to be reliable. If we keep increase Max Leaf Nodes, potentially we can get the accuracy even better, however it reaches the limit of our PC since it takes significantly longer to run codes if we keep increasing.

<p align="center">
Performance Result of Random Forest Model: <br/>
<img src="https://i.imgur.com/3jatzei.png" height="40%" width="60%" alt="Performance Result of Random Forest Model"/>
<br />
</p>

<h2>Building The Engine</h2>

<p align="center">
Songs Recommendation Demonstration: <br/>
<img src="https://i.imgur.com/ueOTU0w.png" height="40%" width="60%" alt="Songs Recommendation Demonstration"/>
<br />
</p>

<h2>Future Considerations</h2>

Although these models are far from perfection, they do have some degree of ability to predict and recommend tracks that is suitable to the input song vibe. Therefore for future studies, the potential future project should explore the following options:
1. <b>Applying PCA</b> prior to training a Decision Tree or Random Forest: This is a dimensionality reduction technique that reduces the correlation between features, which often makes things easier for trees.
2. <b>Increased data access and compute power</b>: By expanding the amount of available information, the accuracy and reliability of our predictions can be improved.
3. <b>Reinforcement Learning coupled with NLP</b>: Shift away from traditional approach and employ a modern technique such as Deep Deterministic Gradient Policy (DDPG) alongside with NLP for linguistic recognition, to model music recommendations as a sequential decision process.


<!--
 ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```
--!>
