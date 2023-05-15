---
layout: post
title:  TSNE Time Series Plots
date:   2023-05-15 13:00:00 +1000
categories: unsupervised visual
---

Understanding the types of data and interactions that exist in multivariate time series data sets can be a challenging 
task. One of the engaging ways I like to explore these datasets is by using an unsupervised ML model named t-SNE. This 
post aims to delve into the logic behind how you can effectively explore time series datasets with this relatively 
straightforward technique.

## TSNE

T-distributed Stochastic Neighbor Embedding or t-SNE is an unsupervised Machine Learning (ML) model. It is often used to 
visualize high-dimensional data in a 2D or 3D space. The model works by first calculating the distance between data points
in an n-dimensional space. It then tries to find a lower dimensional (typically 2 or 3) space where the distributions of
distances between points are preserved. The model is non-linear, meaning it can identify complex relationships between
data points. You can read this [blog post][TSNEBLOG] for a more detailed explanation of how the model works.

## The Dataset

Naturally, a dataset is required for us to explore. For this post, I'll be using simple daily closing stock prices for
Apple, Google, and Amazon. This data will also require some preprocessing. We are not necessarily interested in the 
prices over long periods of time but more so interested in the patterns we see in short time horizon windows. Therefore,
we will take 14-day rolling windows of data as our dataset. This is depicted in the second line of plots. As most people
are likely interested in the relative price movements of the stocks, we will normalize the data by dividing each price by 
the mean of the 14-day window.This is depicted in the third line of plots. There are many other normalization techniques 
that could be used, but this one is simple and effective for this use case.

![Stock Prices]({{ "/assets/images/stock.png" | relative_url }})

## Raw TSNE

With the data windows prepared, it's possible to fit the t-SNE model. First, a t-SNE model object is created. While there
are several parameters that can be tuned, for this example, we will use the defaults which work well enough. The only 
additional thing that needs to be considered is the shape of the data. t-SNE expects a 2D array, so we will reshape our 
data to be one single 42-point array.

{% highlight bash %}
tsne = TSNE(n_components=2)
tsne_data = tsne.fit_transform(normed_data.reshape(-1, 3*14))
{% endhighlight %}

The result is plotted below, with colours ranging from blue to red representing the oldest to the newest windows. It's
interesting to see that there are many points in time that are farther apart being represented as close together. This 
highlights that the model has discovered some patterns in the stock prices that have repeated over time.

![TSNE]({{ "/assets/images/tsne_raw.png" | relative_url }})

## TSNE Time Series Plots
The scatter plot above doesn't really reveal much about the patterns present in the data. To understand what patterns
we're looking at, we would have to identify a specific data point and then plot its data. Perhaps we would do the same 
for a nearby point and then compare the two. This is quite laborious. My solution for this is to slice the data into 
bins across the two dimensions, and assign each time series to its bin. Then we can plot the median of each bin as a 
representation of the approximate behavior this area in the t-SNE reduced dimension represents.

![TSNE]({{ "/assets/images/tsne.png" | relative_url }})

With this representation of the data, we can easily see the various patterns that the three stocks present over time.
There's a significant amount of correlation, as expected, but there are also some unique areas where things aren't that
way. If we wanted to we could now try and identify what is going on in these unique areas or even pinpoint specific times
based on their distance from a window of interest.

## Summary
In summary, exploring multivariate time series datasets can be a complex task. However, using an unsupervised machine 
learning model like t-SNE can simplify the process. This post demonstrated how t-SNE can be used to visualize and 
understand patterns in high-dimensional data, using daily closing stock prices for Apple, Google, and Amazon as an example.

The notebook with the code for this post can be found [here][NOTEBOOK].


[TSNEBLOG]: https://towardsdatascience.com/t-sne-clearly-explained-d84c537f53a
[NOTEBOOK]: https://github.com/GandalfsDad/GandalfsDad.github.io/blob/main/_posts/2023-05-15/tsne.ipynb