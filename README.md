# Airbnb Data EDA and Visualization

Descriptive analysis of Airbnb data from Seattle

# Conclusion

1)  Calendar

    Jan - April : Maintenance

    May - Sep : rental

    Oct - Dec : Take a break

2) Communications affect overall rating and check in rating.

3) Price and Revenue are not affected by ratings.

4) Make rooms for 2 to 3 guests.

5) Providing wireless internet, heating and kitchen are common.

### Data

Find Dataset [here](http://insideairbnb.com/get-the-data.html) and place in input folder by creating it.

### Used libraries:

* [`collections`](https://docs.python.org/3/library/collections.html)
* [`dateutil`](https://dateutil.readthedocs.io/en/stable/)
* [`eli5`](https://pypi.org/project/eli5/)
* [`geopy`](https://pypi.org/project/geopy/)
* [`matplotlib`](https://matplotlib.org/)
* [`nltk`](http://www.nltk.org/)
* [`numpy`](http://www.numpy.org/)
* [`pandas`](https://pandas.pydata.org/)
* [`scipy`](https://www.scipy.org/)
* [`seaborn`](https://seaborn.pydata.org/index.html)
* [`sklearn`](http://scikit-learn.org/stable/index.html)
* [`skopt`](https://scikit-optimize.github.io/)
* [`tqdm`](https://pypi.org/project/tqdm/)
* [`warnings`](https://docs.python.org/3/library/warnings.html)

### Motivation for the project
The aim of the project is to analyze the latest Airbnb data publicly available for three different cities (Seattle), to perform sentiment analysis of the reviews for their customers and to understands main factors responsible for the prise of Airbnb apartments.

### Summary of the results of the analysis

* Overwhelming majority (`> 95%`) of Airbnb reviews are either positive or neutral.
* For all these cities, superhosts tend to have larger total and monthly averaged number of reviews, review scores and yearly availability are larger for superhosts than for ordinary hosts. On the other hand, the number of minimum nights, host response time and the host listings counts are smaller for superhosts than for ordinary hosts. This may reflect the higher popularity of superhosts and their higher level of service, compared to ordinary hosts.
* Among the most important features for daily price predictions are the distance to the city center and the type of the room. However, there are also significant differences between largest influencing features between different cities.
* Based on model trained by the data from different cities, we are able to predict the prices for a given city with a decent [R2 score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html) close to 0.7.
