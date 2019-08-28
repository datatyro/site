# Data Science from Scratch
> You can get this book from [here](https://www.oreilly.com/library/view/data-science-from/9781492041122/).

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

## 22.Network Analysis

- There are [many other notions of centrality](http://en.wikipedia.org/wiki/Centrality) besides the ones we used (although the ones we used are pretty much the most popular ones).
- [NetworkX](http://networkx.github.io/) is a Python library for network analysis. It has functions for computing centralities and for visualizing graphs.
- [Gephi](https://gephi.org/) is a love-it/hate-it GUI-based network visualization tool.

## 23.Recommender Systems

- [Surprise](http://surpriselib.com/) is a Python library for “building and analyzing recommender systems” that seems reasonably popular and up-to-date.
- The [Netflix Prize](http://www.netflixprize.com/) was a somewhat famous competition to build a better system to recommend movies to Netflix users.

## 24. Databases and SQL

- If you’d like to download a relational database to play with, [SQLite](http://www.sqlite.org/) is fast and tiny, while [MySQL](http://www.mysql.com/) and[PostgreSQL](http://www.postgresql.org/) are larger and featureful. All are free and have lots of documentation.
- If you want to explore NoSQL, [MongoDB](http://www.mongodb.org/) is very simple to get started with, which can be both a blessing and somewhat of a curse. It also has pretty good documentation.
- The [Wikipedia article on NoSQL](http://en.wikipedia.org/wiki/NoSQL) almost certainly now contains links to databases that didn’t even exist when this book was written.

## 25.MapReduce

- Like I said, MapReduce feels a lot less popular now than it did when I wrote the first edition. It’s probably not worth investing a ton of your time.
- That said, the most widely used MapReduce system is [Hadoop](http://hadoop.apache.org/). There are various commercial and noncommercial distributions and a huge ecosystem of Hadoop-related tools.
- Amazon.com offers an [Elastic MapReduce](http://aws.amazon.com/elasticmapreduce/) service that’s probably easier than setting up your own cluster.
- Hadoop jobs are typically high-latency, which makes them a poor choice for “real-time” analytics. A popular choice for these workloads is [Spark](http://spark.apache.org/), which can be MapReduce-y.

## 26. Data Ethics

- There is no shortage of people professing important thoughts about data ethics. Searching on Twitter (or your favorite news site) is probably the best way to find out about the most current data ethics controversy.
- If you want something slightly more practical, Mike Loukides, Hilary Mason, and DJ Patil have written a short ebook, [*Ethics and Data Science*](https://www.oreilly.com/library/view/ethics-and-data/9781492043898/), on putting data ethics into practice, which I am honor-bound to recommend on account of Mike being the person who agreed to publish *Data Science from Scratch* way back in 2014. (Exercise: is this ethical of me?)

## 27. Go Forth and Do Data Science

### IPython

I mentioned [IPython](http://ipython.org/) earlier in the book. It provides a shell with far more functionality than the standard Python shell, and it adds “magic functions” that allow you to (among other things) easily copy and paste code (which is normally complicated by the combination of blank lines and whitespace formatting) and run scripts from within the shell.

Mastering IPython will make your life far easier. (Even learning just a little bit of IPython will make your life a lot easier.)

###### NOTE

> In the first edition, I also recommended that you learn about the IPython (now Jupyter) Notebook, a computational environment that allows you to combine text, live Python code, and visualizations.
>
> I’ve since [become a notebook skeptic](https://twitter.com/joelgrus/status/1033035196428378113), as I find that they confuse beginners and encourage bad coding practices. (I have many other reasons too.) You will surely receive plenty of encouragement to use them from people who aren’t me, so just remember that I’m the dissenting voice.

### Mathematics

Throughout this book, we dabbled in linear algebra ([Chapter 4](https://learning.oreilly.com/library/view/data-science-from/9781492041122/ch04.html#linear_algebra)), statistics ([Chapter 5](https://learning.oreilly.com/library/view/data-science-from/9781492041122/ch05.html#statistics)), probability ([Chapter 6](https://learning.oreilly.com/library/view/data-science-from/9781492041122/ch06.html#probability)), and various aspects of machine learning.

To be a good data scientist, you should know much more about these topics, and I encourage you to give each of them a more in-depth study, using the textbooks recommended at the ends of the chapters, your own preferred textbooks, online courses, or even real-life courses.

### Not from Scratch

Implementing things “from scratch” is great for understanding how they work. But it’s generally not great for performance (unless you’re implementing them specifically with performance in mind), ease of use, rapid prototyping, or error handling.

In practice, you’ll want to use well-designed libraries that solidly implement the fundamentals. My original proposal for this book involved a second “now let’s learn the libraries” half that O’Reilly, thankfully, vetoed. Since the first edition came out, Jake VanderPlas has written the [*Python Data Science Handbook*](http://shop.oreilly.com/product/0636920034919.do) (O’Reilly), which is a good introduction to the relevant libraries and would be a good book for you to read next.

### NumPy

[NumPy](http://www.numpy.org/) (for “Numeric Python”) provides facilities for doing “real” scientific computing. It features arrays that perform better than our `list`-vectors, matrices that perform better than our `list`-of-`list`-matrices, and lots of numeric functions for working with them.

NumPy is a building block for many other libraries, which makes it especially valuable to know.

### pandas

[pandas](http://pandas.pydata.org/) provides additional data structures for working with datasets in Python. Its primary abstraction is the `DataFrame`, which is conceptually similar to the NotQuiteABase `Table` class we constructed in [Chapter 24](https://learning.oreilly.com/library/view/data-science-from/9781492041122/ch24.html#databases), but with much more functionality and better performance.

If you’re going to use Python to munge, slice, group, and manipulate datasets, pandas is an invaluable tool.

### scikit-learn

[scikit-learn](http://scikit-learn.org/) is probably the most popular library for doing machine learning in Python. It contains all the models we’ve implemented and many more that we haven’t. On a real problem, you’d never build a decision tree from scratch; you’d let scikit-learn do the heavy lifting. On a real problem, you’d never write an optimization algorithm by hand; you’d count on scikit-learn to already be using a really good one.

Its documentation contains [many, many examples](http://scikit-learn.org/stable/auto_examples/) of what it can do (and, more generally, what machine learning can do).

### Visualization

The matplotlib charts we’ve been creating have been clean and functional but not particularly stylish (and not at all interactive). If you want to get deeper into data visualization, you have several options.

The first is to further explore matplotlib, only a handful of whose features we’ve actually covered. Its website contains many [examples](http://matplotlib.org/examples/) of its functionality and a [gallery](http://matplotlib.org/gallery.html) of some of the more interesting ones. If you want to create static visualizations (say, for printing in a book), this is probably your best next step.

You should also check out [seaborn](https://seaborn.pydata.org/), which is a library that (among other things) makes matplotlib more attractive.

If you’d like to create *interactive* visualizations that you can share on the web, the obvious choice is probably [D3.js](http://d3js.org/), a JavaScript library for creating “data-driven documents” (those are the three Ds). Even if you don’t know much JavaScript, it’s often possible to crib examples from the [D3 gallery](https://github.com/mbostock/d3/wiki/Gallery) and tweak them to work with your data. (Good data scientists copy from the D3 gallery; great data scientists *steal* from the D3 gallery.)

Even if you have no interest in D3, just browsing the gallery is itself a pretty incredible education in data visualization.

[Bokeh](http://bokeh.pydata.org/) is a project that brings D3-style functionality into Python.

### R

Although you can totally get away with not learning [R](http://www.r-project.org/), a lot of data scientists and data science projects use it, so it’s worth getting at least familiar with it.

In part, this is so that you can understand people’s R-based blog posts and examples and code; in part, this is to help you better appreciate the (comparatively) clean elegance of Python; and in part, this is to help you be a more informed participant in the never-ending “R versus Python” flamewars.

### Deep Learning

You can be a data scientist without doing deep learning, but you can’t be a *trendy* data scientist without doing deep learning.

The two most popular deep learning frameworks for Python are [TensorFlow](https://www.tensorflow.org/) (created by Google) and [PyTorch](https://pytorch.org/)(created by Facebook). The internet is full of tutorials for them that range from wonderful to awful.

TensorFlow is older and more widely used, but PyTorch is (in my opinion) much easier to use and (in particular) much more beginner-friendly. I prefer (and recommend) PyTorch, but—as they say—no one ever got fired for choosing TensorFlow.

### Find Data

If you’re doing data science as part of your job, you’ll most likely get the data as part of your job (although not necessarily). What if you’re doing data science for fun? Data is everywhere, but here are some starting points:

- [Data.gov](http://www.data.gov/) is the government’s open data portal. If you want data on anything that has to do with the government (which seems to be most things these days), it’s a good place to start.
- Reddit has a couple of forums, [r/datasets](http://www.reddit.com/r/datasets) and [r/data](http://www.reddit.com/r/data), that are places to both ask for and discover data.
- Amazon.com maintains a collection of [public datasets](http://aws.amazon.com/public-data-sets/) that they’d like you to analyze using their products (but that you can analyze with whatever products you want).
- Robb Seaton has a quirky list of curated datasets [on his blog](http://rs.io/100-interesting-data-sets-for-statistics/).
- [Kaggle](https://www.kaggle.com/) is a site that holds data science competitions. I never managed to get into it (I don’t have much of a competitive nature when it comes to data science), but you might. They host a lot of datasets.
- Google has a newish [Dataset Search](https://toolbox.google.com/datasetsearch) that lets you (you guessed it) search for datasets.

