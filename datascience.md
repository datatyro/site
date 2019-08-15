# Data Science from Scratch
> You can get this book from [there](https://www.oreilly.com/library/view/data-science-from/9781492041122/).

> This page redords the secion of  `Further Exploration Content`


## 2.Python
* There is no shortage of Python tutorials in the world. The [official one](https://docs.python.org/3/tutorial/) is not a bad place to start.

* The [official IPython tutorial](http://ipython.readthedocs.io/en/stable/interactive/index.html) will help you get started with IPython, if you decide to use it. Please use it.

* The [mypy documentation](https://mypy.readthedocs.io/en/stable/) will tell you more than you ever wanted to know about Python type annotations and type checking.

## 3.Visualizing Data
* The [matplotlib Gallery](https://matplotlib.org/gallery.html) will give you a good idea of the sorts of things you can do with matplotlib (and how to do them).

* [seaborn](https://seaborn.pydata.org/) is built on top of matplotlib and allows you to easily produce prettier (and more complex) visualizations.

* [Altair](https://altair-viz.github.io/) is a newer Python library for creating declarative visualizations.

* [D3.js](http://d3js.org/) is a JavaScript library for producing sophisticated interactive visualizations for the web. Although it is not in Python, it is widely used, and it is well worth your while to be familiar with it.

* [Bokeh](http://bokeh.pydata.org/) is a library that brings D3-style visualizations into Python.

## 4.Linear Algebra
* Linear algebra is widely used by data scientists (frequently implicitly, and not infrequently by people who don’t understand it). It wouldn’t be a bad idea to read a textbook. You can find several freely available online:

    * [Linear Algebra](http://joshua.smcvt.edu/linearalgebra/), by Jim Hefferon (Saint Michael’s College)

    * [Linear Algebra](https://www.math.ucdavis.edu/~linear/linear-guest.pdf), by David Cherney, Tom Denton, Rohit Thomas, and Andrew Waldron (UC Davis)

    * If you are feeling adventurous, [Linear Algebra Done Wrong](https://www.math.brown.edu/~treil/papers/LADW/LADW_2017-09-04.pdf), by Sergei Treil (Brown University), is a more advanced introduction.

* All of the machinery we built in this chapter you get for free if you use [NumPy](http://www.numpy.org/). (You get a lot more too, including much better performance.)

## 5.Statistics
* [SciPy](https://www.scipy.org/), [pandas](http://pandas.pydata.org/), and [StatsModels](http://www.statsmodels.org/) all come with a wide variety of statistical functions.

* Statistics is *important*. (Or maybe statistics are important?) If you want to be a better data scientist, it would be a good idea to read a statistics textbook. Many are freely available online, including:

    * [Introductory Statistics](https://open.umn.edu/opentextbooks/textbooks/introductory-statistics), by Douglas Shafer and Zhiyi Zhang (Saylor Foundation)

    * [OnlineStatBook](http://onlinestatbook.com/), by David Lane (Rice University)

    * [Introductory Statistics](https://openstax.org/details/introductory-statistics), by OpenStax (OpenStax College)

## 6.Probability
* [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html) contains PDF and CDF functions for most of the popular probability distributions.

* Remember how, at the end of Chapter 5, I said that it would be a good idea to study a statistics textbook? It would also be a good idea to study a probability textbook. The best one I know that’s available online is [Introduction to Probability](http://www.dartmouth.edu/~chance/teaching_aids/books_articles/probability_book/book.html), by Charles M. Grinstead and J. Laurie Snell (American Mathematical Society).

## 7.Hypothesis and Inference
* We’ve barely scratched the surface of what you should know about statistical inference. The books recommended at the end of Chapter 5 go into a lot more detail.

* Coursera offers a [Data Analysis and Statistical Inference](https://www.coursera.org/course/statistics) course that covers many of these topics.

## 8.Gradient Descent
* At this point, you’re undoubtedly sick of me recommending that you read textbooks. If it’s any consolation, [Active Calculus 1.0](https://scholarworks.gvsu.edu/books/10/), by Matthew Boelkins, David Austin, and Steven Schlicker (Grand Valley State University Libraries), seems nicer than the calculus textbooks I learned from.

* Sebastian Ruder has an [epic blog post](http://ruder.io/optimizing-gradient-descent/index.html) comparing gradient descent and its many variants.

## 9.Getting Data
* [pandas](http://pandas.pydata.org/) is the primary library that data science types use for working with—and, in particular, importing—data.

* [Scrapy](http://scrapy.org/) is a full-featured library for building complicated web scrapers that do things like follow unknown links.

* [Kaggle](https://www.kaggle.com/datasets) hosts a large collection of datasets.

## 10.Working with Data
* As mentioned at the end of Chapter 9, pandas is probably the primary Python tool for cleaning, munging, manipulating, and working with data. All the examples we did by hand in this chapter could be done much more simply using pandas. [Python for Data Analysis](https://learning.oreilly.com/library/view/python-for-data/9781491957653/) (O’Reilly), by Wes McKinney, is probably the best way to learn pandas.

* scikit-learn has a wide variety of [matrix decomposition](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition) functions, including PCA.

## 11.Machine Learning
* The Coursera [Machine Learning](https://www.coursera.org/course/ml) course is the original MOOC and is a good place to get a deeper understanding of the basics of machine learning.

* *The Elements of Statistical Learning*, by Jerome H. Friedman, Robert Tibshirani, and Trevor Hastie (Springer), is a somewhat canonical textbook that can be [downloaded online for free](http://stanford.io/1ycOXbo). But be warned: it’s very mathy.

## 12.k-Nearest Neighbors
* scikit-learn has many [nearest neighbor](https://scikit-learn.org/stable/modules/neighbors.html) models.

## 13.Naive Bayes
* Paul Graham’s articles “[A Plan for Spam](http://www.paulgraham.com/spam.html)” and “[Better Bayesian Filtering](http://www.paulgraham.com/better.html)” are interesting and give more insight into the ideas behind building spam filters.

* [scikit-learn](https://scikit-learn.org/stable/modules/naive_bayes.html) contains a **BernoulliNB** model that implements the same Naive Bayes algorithm we implemented here, as well as other variations on the model.

## 15.Multiple Regression
* Regression has a rich and expansive theory behind it. This is another place where you should consider reading a textbook, or at least a lot of Wikipedia articles.

* scikit-learn has a [linear_model module](https://scikit-learn.org/stable/modules/linear_model.html) that provides a LinearRegression model similar to ours, as well as ridge regression, lasso regression, and other types of regularization.

* [Statsmodels](https://www.statsmodels.org/) is another Python module that contains (among other things) linear regression models.

## 16.Logistic Regression
* scikit-learn has modules for both [logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) and [support vector machines](https://scikit-learn.org/stable/modules/svm.html).

* [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) is the support vector machine implementation that scikit-learn is using behind the scenes. Its website has a variety of useful documentation about support vector machines.

## 17.Decision Trees
* scikit-learn has many [decision tree](https://scikit-learn.org/stable/modules/tree.html) models. It also has an [ensemble](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble) module that includes a **RandomForestClassifier** as well as other ensemble methods.

* [XGBoost](https://xgboost.ai/) is a library for training *gradient boosted* decision trees that tends to win a lot of Kaggle-style machine learning competitions.

* We’ve barely scratched the surface of decision trees and their algorithms. [Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning) is a good starting point for broader exploration.

## 18.Neural Networks
* My blog post on “[Fizz Buzz in Tensorflow](http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/)” is pretty good.

## 19.Deep Learning
Deep learning is really hot right now, and in this chapter we barely scratched its surface. There are many good books and blog posts (and many, many bad blog posts) about almost any aspect of deep learning you’d like to know about.


   * The canonical textbook [Deep Learning](https://www.deeplearningbook.org/), by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (MIT Press), is freely available online. It is very good, but it involves quite a bit of mathematics.

    * Francois Chollet’s [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python) (Manning) is a great introduction to the Keras library, after which our deep learning library is sort of patterned.

    * I myself mostly use [PyTorch](https://pytorch.org/) for deep learning. Its website has lots of documentation and tutorials.

## 20.Clustering
* scikit-learn has an entire module, [sklearn.cluster](http://scikit-learn.org/stable/modules/clustering.html), that contains several clustering algorithms including **KMeans** and the Ward hierarchical clustering algorithm (which uses a different criterion for merging clusters than ours did).

* [SciPy](http://www.scipy.org/) has two clustering models: scipy.cluster.vq, which does k-means, and scipy.cluster.hierarchy, which has a variety of hierarchical clustering algorithms.

## 21.Natural Language Processing
* [NLTK](http://www.nltk.org/) is a popular library of NLP tools for Python. It has its own entire [book](http://www.nltk.org/book/), which is available to read online.

- [gensim](http://radimrehurek.com/gensim/) is a Python library for topic modeling, which is a better bet than our from-scratch model.

- [spaCy](https://spacy.io/) is a library for “Industrial Strength Natural Language Processing in Python” and is also quite popular.
- Andrej Karpathy has a famous blog post, “[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)”, that’s very much worth reading.

- My day job involves building [AllenNLP](https://allennlp.org/), a Python library for doing NLP research. (At least, as of the time this book went to press, it did.) The library is quite beyond the scope of this book, but you might still find it interesting, and it has a cool interactive demo of many state-of-the-art NLP models.

