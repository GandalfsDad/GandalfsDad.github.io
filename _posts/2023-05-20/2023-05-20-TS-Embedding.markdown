---
layout: post
title:  Time Series Embedding
date:   2023-05-20 13:00:00 +1000
categories: embedding 
---

Time series data is always fascinating to me. A recent idea I had was to use embedding techniques in order to speed up 
the identification of rare events in data. Historically a lot of my work has used techniques like K Means clustering
or t-SNE to identify patterns in data. However, these techniques aren't always great at working with rare events.
This post explores some of the steps i used to embed time series data.

## The Dataset
The dataset I used for this example is some weather data measured in Banladesh that is available on [Kaggle][KAGGLE].
It's not a very large dataset but is enough to demonstrate the approach. I've selected 4 columns of data to work with:
- temperature
- dewpoint
- humidity
- pressure

![Weather]({{ "/assets/images/weather.png" | relative_url }})

As far as normalization is concerned I have simply divided all values by their respective maxima. This is simple but 
quite effective. An observation I have made is that when working with other datasets, it is often useful to normalise
per attribute in different ways. For example some % style values it simply makes sense to divide by 100. Other values you 
may have no interest in absolute values and only care about the relative values. In this case you may want to divide by the
max value within your time window.

The data is measures in 30min intervals so I have taken Daily Windows of the data. This generates a 48 x 4 matrix for
each `observation`.


## Embedding
The modeling structure i've used for this embedding is very much like an image embedding. My original idea came from the
thought that the Convolutional layers would be able to extract features that roughly correspond to interactions between
parameters in the time series data. This is an interesting use of CNNs that I don't think is often discussed. Below is an 
excerpt from the model definition. The full code can be found in this [repo][REPO]. 

{% highlight python %}
def __buildEncoder(self):
    mp2c, mp2h, mp2w = self.__flatmp2_dim()

    self.encoder = nn.Sequential(
        # Input shape: (batch_size, 1, t, f)
        nn.Conv2d(1, self.__c1_channels, self.__c1_kernel_size, stride=self.__c1_stride, padding=self.__c1_padding),
        nn.Dropout(self.__dropout),
        nn.ReLU(),
        nn.MaxPool2d(self.__mp1_kernel_size),
        nn.Conv2d(self.__c1_channels, self.__c2_channels, self.__c2_kernel_size, stride=self.__c2_stride, padding=self.__c2_padding),
        nn.Dropout(self.__dropout),
        nn.ReLU(),
        nn.MaxPool2d(self.__mp2_kernel_size),
        nn.Flatten(),
        nn.Linear(mp2c*mp2h*mp2w,self.__hidden_dim),
        nn.Dropout(self.__dropout),
        nn.Linear(self.__hidden_dim, self.__embed_dim),
        # Output shape: (batch_size, embed_dim)
    )
{% endhighlight %}

## Training
The

[KAGGLE]: https://www.kaggle.com/datasets/talhabu/bangladesh-weather-history?resource=download
[REPO]: https://www.githubuvjytcjy