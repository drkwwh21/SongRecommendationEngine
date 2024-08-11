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

<h2>Data Preparation and EDA:</h2>

The data source used created by Spotify and extracted via API. It was posted as part of a [Kaggle project](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks?select=artists.csv) and is composed of two datasets - Tracks.csv and Artist.csv, used to train and develop our model. Tracks.csv contains 587k unique values and has 20 features that span from numerical, ordinal and categorical. This data frame had no null-values which was not surprising as it was directly provided by Spotify, but also shocking, as generally some data cleansing is required.

<h2>Methodology:</h2>

<p align="center">
Pipelines and Workflow: <br/>
<img src="https://i.imgur.com/1Mvu9hr.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
</p>

The Spotify data set does not contain a target value appropriate for our objective, i.e. there is no
prediction target. Hence, we first need to cluster the songs (X variable) into groups and the resultant
group number will be its target value (Y value). This will be done using an unsupervised technique,
K-Means Clustering.
<br />
<br />
Once X and Y are built, we now have a classification problem, i.e. the data set contains feature
variables, X and target variables, Y. Given a new song, classify it into the one of the cluster Y based
on the similarities of the songs' features. This can be done by using a myriad of supervised learning
models. The following methods will be used to train, finetune and ultimately choosing the best
classification model for our objective:
- K-Nearest Neighbours
- Decision Tree
- Random Forest

<!--
 ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```
--!>
