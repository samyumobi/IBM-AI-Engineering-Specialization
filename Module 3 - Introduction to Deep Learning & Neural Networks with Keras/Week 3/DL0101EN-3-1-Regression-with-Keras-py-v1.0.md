<a href="https://cognitiveclass.ai"><img src = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/Logos/organization_logo/organization_logo.png" width = 400> </a>

<h1 align=center><font size = 5>Regression Models with Keras</font></h1>

## Introduction

As we discussed in the videos, despite the popularity of more powerful libraries such as PyToch and TensorFlow, they are not easy to use and have a steep learning curve. So, for people who are just starting to learn deep learning, there is no better library to use other than the Keras library. 

Keras is a high-level API for building deep learning models. It has gained favor for its ease of use and syntactic simplicity facilitating fast development. As you will see in this lab and the other labs in this course, building a very complex deep learning network can be achieved with Keras with only few lines of code. You will appreciate Keras even more, once you learn how to build deep models using PyTorch and TensorFlow in the other courses.

So, in this lab, you will learn how to use the Keras library to build a regression model.

## Table of Contents

<div class="alert alert-block alert-info" style="margin-top: 20px">

<font size = 3>
    
1. <a href="#item31">Download and Clean Dataset</a>  
2. <a href="#item32">Import Keras</a>  
3. <a href="#item33">Build a Neural Network</a>  
4. <a href="#item34">Train and Test the Network</a>  

</font>
</div>

<a id="item31"></a>

## Download and Clean Dataset

Let's start by importing the <em>pandas</em> and the Numpy libraries.


```python
import pandas as pd
import numpy as np
```

We will be playing around with the same dataset that we used in the videos.

<strong>The dataset is about the compressive strength of different samples of concrete based on the volumes of the different ingredients that were used to make them. Ingredients include:</strong>

<strong>1. Cement</strong>

<strong>2. Blast Furnace Slag</strong>

<strong>3. Fly Ash</strong>

<strong>4. Water</strong>

<strong>5. Superplasticizer</strong>

<strong>6. Coarse Aggregate</strong>

<strong>7. Fine Aggregate</strong>

Let's download the data and read it into a <em>pandas</em> dataframe.


```python
concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cement</th>
      <th>Blast Furnace Slag</th>
      <th>Fly Ash</th>
      <th>Water</th>
      <th>Superplasticizer</th>
      <th>Coarse Aggregate</th>
      <th>Fine Aggregate</th>
      <th>Age</th>
      <th>Strength</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1040.0</td>
      <td>676.0</td>
      <td>28</td>
      <td>79.99</td>
    </tr>
    <tr>
      <th>1</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1055.0</td>
      <td>676.0</td>
      <td>28</td>
      <td>61.89</td>
    </tr>
    <tr>
      <th>2</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>270</td>
      <td>40.27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>365</td>
      <td>41.05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>198.6</td>
      <td>132.4</td>
      <td>0.0</td>
      <td>192.0</td>
      <td>0.0</td>
      <td>978.4</td>
      <td>825.5</td>
      <td>360</td>
      <td>44.30</td>
    </tr>
  </tbody>
</table>
</div>



So the first concrete sample has 540 cubic meter of cement, 0 cubic meter of blast furnace slag, 0 cubic meter of fly ash, 162 cubic meter of water, 2.5 cubic meter of superplaticizer, 1040 cubic meter of coarse aggregate, 676 cubic meter of fine aggregate. Such a concrete mix which is 28 days old, has a compressive strength of 79.99 MPa. 

#### Let's check how many data points we have.


```python
concrete_data.shape
```




    (1030, 9)



So, there are approximately 1000 samples to train our model on. Because of the few samples, we have to be careful not to overfit the training data.

Let's check the dataset for any missing values.


```python
concrete_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cement</th>
      <th>Blast Furnace Slag</th>
      <th>Fly Ash</th>
      <th>Water</th>
      <th>Superplasticizer</th>
      <th>Coarse Aggregate</th>
      <th>Fine Aggregate</th>
      <th>Age</th>
      <th>Strength</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1030.000000</td>
      <td>1030.000000</td>
      <td>1030.000000</td>
      <td>1030.000000</td>
      <td>1030.000000</td>
      <td>1030.000000</td>
      <td>1030.000000</td>
      <td>1030.000000</td>
      <td>1030.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>281.167864</td>
      <td>73.895825</td>
      <td>54.188350</td>
      <td>181.567282</td>
      <td>6.204660</td>
      <td>972.918932</td>
      <td>773.580485</td>
      <td>45.662136</td>
      <td>35.817961</td>
    </tr>
    <tr>
      <th>std</th>
      <td>104.506364</td>
      <td>86.279342</td>
      <td>63.997004</td>
      <td>21.354219</td>
      <td>5.973841</td>
      <td>77.753954</td>
      <td>80.175980</td>
      <td>63.169912</td>
      <td>16.705742</td>
    </tr>
    <tr>
      <th>min</th>
      <td>102.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>121.800000</td>
      <td>0.000000</td>
      <td>801.000000</td>
      <td>594.000000</td>
      <td>1.000000</td>
      <td>2.330000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>192.375000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>164.900000</td>
      <td>0.000000</td>
      <td>932.000000</td>
      <td>730.950000</td>
      <td>7.000000</td>
      <td>23.710000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>272.900000</td>
      <td>22.000000</td>
      <td>0.000000</td>
      <td>185.000000</td>
      <td>6.400000</td>
      <td>968.000000</td>
      <td>779.500000</td>
      <td>28.000000</td>
      <td>34.445000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>350.000000</td>
      <td>142.950000</td>
      <td>118.300000</td>
      <td>192.000000</td>
      <td>10.200000</td>
      <td>1029.400000</td>
      <td>824.000000</td>
      <td>56.000000</td>
      <td>46.135000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>540.000000</td>
      <td>359.400000</td>
      <td>200.100000</td>
      <td>247.000000</td>
      <td>32.200000</td>
      <td>1145.000000</td>
      <td>992.600000</td>
      <td>365.000000</td>
      <td>82.600000</td>
    </tr>
  </tbody>
</table>
</div>




```python
concrete_data.isnull().sum()
```




    Cement                0
    Blast Furnace Slag    0
    Fly Ash               0
    Water                 0
    Superplasticizer      0
    Coarse Aggregate      0
    Fine Aggregate        0
    Age                   0
    Strength              0
    dtype: int64



The data looks very clean and is ready to be used to build our model.

#### Split data into predictors and target

The target variable in this problem is the concrete sample strength. Therefore, our predictors will be all the other columns.


```python
concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column
```

<a id="item2"></a>

Let's do a quick sanity check of the predictors and the target dataframes.


```python
predictors.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cement</th>
      <th>Blast Furnace Slag</th>
      <th>Fly Ash</th>
      <th>Water</th>
      <th>Superplasticizer</th>
      <th>Coarse Aggregate</th>
      <th>Fine Aggregate</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1040.0</td>
      <td>676.0</td>
      <td>28</td>
    </tr>
    <tr>
      <th>1</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1055.0</td>
      <td>676.0</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>270</td>
    </tr>
    <tr>
      <th>3</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>365</td>
    </tr>
    <tr>
      <th>4</th>
      <td>198.6</td>
      <td>132.4</td>
      <td>0.0</td>
      <td>192.0</td>
      <td>0.0</td>
      <td>978.4</td>
      <td>825.5</td>
      <td>360</td>
    </tr>
  </tbody>
</table>
</div>




```python
target.head()
```




    0    79.99
    1    61.89
    2    40.27
    3    41.05
    4    44.30
    Name: Strength, dtype: float64



Finally, the last step is to normalize the data by substracting the mean and dividing by the standard deviation.


```python
predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cement</th>
      <th>Blast Furnace Slag</th>
      <th>Fly Ash</th>
      <th>Water</th>
      <th>Superplasticizer</th>
      <th>Coarse Aggregate</th>
      <th>Fine Aggregate</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.476712</td>
      <td>-0.856472</td>
      <td>-0.846733</td>
      <td>-0.916319</td>
      <td>-0.620147</td>
      <td>0.862735</td>
      <td>-1.217079</td>
      <td>-0.279597</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.476712</td>
      <td>-0.856472</td>
      <td>-0.846733</td>
      <td>-0.916319</td>
      <td>-0.620147</td>
      <td>1.055651</td>
      <td>-1.217079</td>
      <td>-0.279597</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.491187</td>
      <td>0.795140</td>
      <td>-0.846733</td>
      <td>2.174405</td>
      <td>-1.038638</td>
      <td>-0.526262</td>
      <td>-2.239829</td>
      <td>3.551340</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.491187</td>
      <td>0.795140</td>
      <td>-0.846733</td>
      <td>2.174405</td>
      <td>-1.038638</td>
      <td>-0.526262</td>
      <td>-2.239829</td>
      <td>5.055221</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.790075</td>
      <td>0.678079</td>
      <td>-0.846733</td>
      <td>0.488555</td>
      <td>-1.038638</td>
      <td>0.070492</td>
      <td>0.647569</td>
      <td>4.976069</td>
    </tr>
  </tbody>
</table>
</div>



Let's save the number of predictors to *n_cols* since we will need this number when building our network.


```python
predictors_norm.shape
```




    (1030, 8)




```python
n_cols = predictors_norm.shape[1] # number of predictors
```

<a id="item1"></a>

<a id='item32'></a>

## Import Keras

Recall from the videos that Keras normally runs on top of a low-level library such as TensorFlow. This means that to be able to use the Keras library, you will have to install TensorFlow first and when you import the Keras library, it will be explicitly displayed what backend was used to install the Keras library. In CC Labs, we used TensorFlow as the backend to install Keras, so it should clearly print that when we import Keras.

#### Let's go ahead and import the Keras library


```python
import keras
```

    Using TensorFlow backend.
    /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:519: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:520: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:521: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:522: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:523: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:528: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])


As you can see, the TensorFlow backend was used to install the Keras library.

Let's import the rest of the packages from the Keras library that we will need to build our regressoin model.


```python
from keras.models import Sequential
from keras.layers import Dense
```

<a id='item33'></a>

## Build a Neural Network

Let's define a function that defines our regression model for us so that we can conveniently call it to create our model.


```python
# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
```

The above function create a model that has two hidden layers, each of 50 hidden units.

<a id="item4"></a>

<a id='item34'></a>

## Train and Test the Network

Let's call the function now to create our model.


```python
# build the model
model = regression_model()
```

Next, we will train and test the model at the same time using the *fit* method. We will leave out 30% of the data for validation and we will train the model for 100 epochs.


```python
# fit the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)
```

    Train on 721 samples, validate on 309 samples
    Epoch 1/100
     - 1s - loss: 1642.8973 - val_loss: 1138.6404
    Epoch 2/100
     - 0s - loss: 1491.0381 - val_loss: 1003.0422
    Epoch 3/100
     - 0s - loss: 1246.3662 - val_loss: 793.3029
    Epoch 4/100
     - 0s - loss: 889.8913 - val_loss: 532.3963
    Epoch 5/100
     - 0s - loss: 522.1839 - val_loss: 309.6038
    Epoch 6/100
     - 0s - loss: 298.1172 - val_loss: 206.4897
    Epoch 7/100
     - 0s - loss: 239.0344 - val_loss: 183.4244
    Epoch 8/100
     - 0s - loss: 224.8695 - val_loss: 178.0423
    Epoch 9/100
     - 0s - loss: 212.3456 - val_loss: 174.6256
    Epoch 10/100
     - 0s - loss: 203.2201 - val_loss: 172.2576
    Epoch 11/100
     - 0s - loss: 194.5847 - val_loss: 168.3436
    Epoch 12/100
     - 0s - loss: 187.9129 - val_loss: 164.9683
    Epoch 13/100
     - 0s - loss: 182.6424 - val_loss: 164.1041
    Epoch 14/100
     - 0s - loss: 177.8551 - val_loss: 162.1620
    Epoch 15/100
     - 0s - loss: 174.2737 - val_loss: 160.6915
    Epoch 16/100
     - 0s - loss: 169.6960 - val_loss: 158.6234
    Epoch 17/100
     - 0s - loss: 166.9922 - val_loss: 159.0966
    Epoch 18/100
     - 0s - loss: 163.9187 - val_loss: 154.8786
    Epoch 19/100
     - 0s - loss: 161.1244 - val_loss: 155.5885
    Epoch 20/100
     - 0s - loss: 159.1159 - val_loss: 155.8804
    Epoch 21/100
     - 0s - loss: 156.4945 - val_loss: 155.1282
    Epoch 22/100
     - 0s - loss: 154.7290 - val_loss: 152.3714
    Epoch 23/100
     - 0s - loss: 152.6718 - val_loss: 154.7087
    Epoch 24/100
     - 0s - loss: 150.8728 - val_loss: 154.5976
    Epoch 25/100
     - 0s - loss: 148.8314 - val_loss: 155.0684
    Epoch 26/100
     - 0s - loss: 147.4073 - val_loss: 150.9646
    Epoch 27/100
     - 0s - loss: 145.2061 - val_loss: 154.9444
    Epoch 28/100
     - 0s - loss: 143.2990 - val_loss: 153.8191
    Epoch 29/100
     - 0s - loss: 141.6315 - val_loss: 152.1036
    Epoch 30/100
     - 0s - loss: 139.1372 - val_loss: 151.6358
    Epoch 31/100
     - 0s - loss: 137.9569 - val_loss: 152.9613
    Epoch 32/100
     - 0s - loss: 136.4223 - val_loss: 152.3521
    Epoch 33/100
     - 0s - loss: 134.8102 - val_loss: 155.1782
    Epoch 34/100
     - 0s - loss: 132.9896 - val_loss: 154.5505
    Epoch 35/100
     - 0s - loss: 131.1887 - val_loss: 154.5572
    Epoch 36/100
     - 0s - loss: 129.0360 - val_loss: 154.4008
    Epoch 37/100
     - 0s - loss: 127.4198 - val_loss: 156.5851
    Epoch 38/100
     - 0s - loss: 126.1008 - val_loss: 154.8160
    Epoch 39/100
     - 0s - loss: 124.1669 - val_loss: 155.8648
    Epoch 40/100
     - 0s - loss: 121.8989 - val_loss: 155.9319
    Epoch 41/100
     - 0s - loss: 119.8627 - val_loss: 156.9425
    Epoch 42/100
     - 0s - loss: 118.1177 - val_loss: 155.5383
    Epoch 43/100
     - 0s - loss: 115.7162 - val_loss: 156.7435
    Epoch 44/100
     - 0s - loss: 113.5165 - val_loss: 157.0347
    Epoch 45/100
     - 0s - loss: 112.0501 - val_loss: 156.5068
    Epoch 46/100
     - 0s - loss: 108.9040 - val_loss: 161.2216
    Epoch 47/100
     - 0s - loss: 105.7642 - val_loss: 158.7604
    Epoch 48/100
     - 0s - loss: 103.1494 - val_loss: 158.5477
    Epoch 49/100
     - 0s - loss: 100.4027 - val_loss: 164.1418
    Epoch 50/100
     - 0s - loss: 98.2303 - val_loss: 160.4035
    Epoch 51/100
     - 0s - loss: 95.3572 - val_loss: 164.2010
    Epoch 52/100
     - 0s - loss: 92.1647 - val_loss: 159.4353
    Epoch 53/100
     - 0s - loss: 88.9346 - val_loss: 160.2736
    Epoch 54/100
     - 0s - loss: 85.8910 - val_loss: 158.1121
    Epoch 55/100
     - 0s - loss: 83.1893 - val_loss: 162.9087
    Epoch 56/100
     - 0s - loss: 80.1870 - val_loss: 153.5208
    Epoch 57/100
     - 0s - loss: 77.6929 - val_loss: 160.4834
    Epoch 58/100
     - 0s - loss: 75.4049 - val_loss: 156.4961
    Epoch 59/100
     - 0s - loss: 72.4659 - val_loss: 153.7176
    Epoch 60/100
     - 0s - loss: 71.2193 - val_loss: 149.8984
    Epoch 61/100
     - 0s - loss: 68.3378 - val_loss: 155.8123
    Epoch 62/100
     - 0s - loss: 66.5440 - val_loss: 151.2405
    Epoch 63/100
     - 0s - loss: 64.3276 - val_loss: 151.7885
    Epoch 64/100
     - 0s - loss: 62.7412 - val_loss: 150.2526
    Epoch 65/100
     - 0s - loss: 61.2374 - val_loss: 147.0923
    Epoch 66/100
     - 0s - loss: 59.3588 - val_loss: 148.3289
    Epoch 67/100
     - 0s - loss: 58.0182 - val_loss: 153.0730
    Epoch 68/100
     - 0s - loss: 56.9006 - val_loss: 151.2340
    Epoch 69/100
     - 0s - loss: 55.6212 - val_loss: 139.1614
    Epoch 70/100
     - 0s - loss: 54.8000 - val_loss: 145.5900
    Epoch 71/100
     - 0s - loss: 53.1532 - val_loss: 146.4986
    Epoch 72/100
     - 0s - loss: 52.1182 - val_loss: 137.4450
    Epoch 73/100
     - 0s - loss: 51.5304 - val_loss: 146.2266
    Epoch 74/100
     - 0s - loss: 50.2355 - val_loss: 143.0448
    Epoch 75/100
     - 0s - loss: 49.1728 - val_loss: 141.0138
    Epoch 76/100
     - 0s - loss: 48.4387 - val_loss: 143.1185
    Epoch 77/100
     - 0s - loss: 47.7076 - val_loss: 144.4417
    Epoch 78/100
     - 0s - loss: 47.9180 - val_loss: 144.1942
    Epoch 79/100
     - 0s - loss: 46.5495 - val_loss: 139.5254
    Epoch 80/100
     - 0s - loss: 45.6173 - val_loss: 138.5983
    Epoch 81/100
     - 0s - loss: 45.2967 - val_loss: 135.7176
    Epoch 82/100
     - 0s - loss: 44.6447 - val_loss: 140.7662
    Epoch 83/100
     - 0s - loss: 43.7236 - val_loss: 132.9101
    Epoch 84/100
     - 0s - loss: 43.3447 - val_loss: 132.1793
    Epoch 85/100
     - 0s - loss: 42.6491 - val_loss: 146.1069
    Epoch 86/100
     - 0s - loss: 42.4216 - val_loss: 141.6966
    Epoch 87/100
     - 0s - loss: 41.8709 - val_loss: 129.4120
    Epoch 88/100
     - 1s - loss: 41.3602 - val_loss: 137.9095
    Epoch 89/100
     - 0s - loss: 40.5608 - val_loss: 132.8454
    Epoch 90/100
     - 0s - loss: 40.2932 - val_loss: 143.9480
    Epoch 91/100
     - 0s - loss: 40.2785 - val_loss: 133.2923
    Epoch 92/100
     - 0s - loss: 39.8243 - val_loss: 137.3132
    Epoch 93/100
     - 0s - loss: 39.3652 - val_loss: 130.1561
    Epoch 94/100
     - 0s - loss: 38.6898 - val_loss: 138.9091
    Epoch 95/100
     - 0s - loss: 38.1383 - val_loss: 134.7223
    Epoch 96/100
     - 0s - loss: 38.3740 - val_loss: 139.8303
    Epoch 97/100
     - 0s - loss: 37.8689 - val_loss: 134.9660
    Epoch 98/100
     - 0s - loss: 37.8914 - val_loss: 146.2875
    Epoch 99/100
     - 0s - loss: 37.5317 - val_loss: 138.4955
    Epoch 100/100
     - 0s - loss: 36.7916 - val_loss: 138.1028





    <keras.callbacks.History at 0x7fa4eeafda20>



<strong>You can refer to this [link](https://keras.io/models/sequential/) to learn about other functions that you can use for prediction or evaluation.</strong>

Feel free to vary the following and note what impact each change has on the model's performance:

1. Increase or decreate number of neurons in hidden layers
2. Add more hidden layers
3. Increase number of epochs


```python
from keras.regularizers import l2
```


```python
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
```


```python
model = regression_model()
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)
```

    Train on 721 samples, validate on 309 samples
    Epoch 1/100
     - 2s - loss: 1678.5466 - val_loss: 1147.0707
    Epoch 2/100
     - 0s - loss: 1238.1736 - val_loss: 283.7827
    Epoch 3/100
     - 0s - loss: 374.7241 - val_loss: 209.8272
    Epoch 4/100
     - 1s - loss: 244.6234 - val_loss: 178.6160
    Epoch 5/100
     - 0s - loss: 213.3278 - val_loss: 171.0222
    Epoch 6/100
     - 0s - loss: 192.7602 - val_loss: 170.0819
    Epoch 7/100
     - 0s - loss: 177.2492 - val_loss: 167.3894
    Epoch 8/100
     - 0s - loss: 165.3635 - val_loss: 176.6982
    Epoch 9/100
     - 0s - loss: 147.8917 - val_loss: 155.8447
    Epoch 10/100
     - 0s - loss: 135.8881 - val_loss: 151.0703
    Epoch 11/100
     - 1s - loss: 125.3029 - val_loss: 162.7557
    Epoch 12/100
     - 0s - loss: 110.5251 - val_loss: 143.7503
    Epoch 13/100
     - 1s - loss: 101.0693 - val_loss: 164.5232
    Epoch 14/100
     - 0s - loss: 86.3713 - val_loss: 136.9956
    Epoch 15/100
     - 0s - loss: 79.1094 - val_loss: 132.9357
    Epoch 16/100
     - 0s - loss: 70.8154 - val_loss: 128.1131
    Epoch 17/100
     - 0s - loss: 62.7189 - val_loss: 154.2152
    Epoch 18/100
     - 0s - loss: 58.2721 - val_loss: 120.2661
    Epoch 19/100
     - 0s - loss: 56.1031 - val_loss: 146.2321
    Epoch 20/100
     - 0s - loss: 53.3201 - val_loss: 114.6441
    Epoch 21/100
     - 0s - loss: 49.8098 - val_loss: 135.5021
    Epoch 22/100
     - 1s - loss: 48.2499 - val_loss: 107.8846
    Epoch 23/100
     - 0s - loss: 45.7272 - val_loss: 114.1716
    Epoch 24/100
     - 0s - loss: 41.7172 - val_loss: 103.9725
    Epoch 25/100
     - 0s - loss: 41.8904 - val_loss: 104.1277
    Epoch 26/100
     - 0s - loss: 38.6644 - val_loss: 116.5783
    Epoch 27/100
     - 0s - loss: 38.5613 - val_loss: 136.7615
    Epoch 28/100
     - 0s - loss: 37.4106 - val_loss: 115.2337
    Epoch 29/100
     - 0s - loss: 34.3187 - val_loss: 98.7548
    Epoch 30/100
     - 0s - loss: 32.9882 - val_loss: 95.6532
    Epoch 31/100
     - 0s - loss: 33.5868 - val_loss: 93.5866
    Epoch 32/100
     - 1s - loss: 34.2834 - val_loss: 115.1779
    Epoch 33/100
     - 0s - loss: 33.3460 - val_loss: 98.4606
    Epoch 34/100
     - 1s - loss: 30.3309 - val_loss: 101.4010
    Epoch 35/100
     - 1s - loss: 28.2946 - val_loss: 120.1226
    Epoch 36/100
     - 1s - loss: 27.9271 - val_loss: 105.4133
    Epoch 37/100
     - 0s - loss: 29.8858 - val_loss: 99.0975
    Epoch 38/100
     - 0s - loss: 26.0860 - val_loss: 91.8538
    Epoch 39/100
     - 0s - loss: 25.8513 - val_loss: 93.3325
    Epoch 40/100
     - 0s - loss: 24.4484 - val_loss: 90.5172
    Epoch 41/100
     - 0s - loss: 24.1827 - val_loss: 105.6903
    Epoch 42/100
     - 1s - loss: 25.7052 - val_loss: 88.3977
    Epoch 43/100
     - 0s - loss: 22.2714 - val_loss: 88.0908
    Epoch 44/100
     - 0s - loss: 23.1589 - val_loss: 89.4508
    Epoch 45/100
     - 0s - loss: 22.1152 - val_loss: 79.5473
    Epoch 46/100
     - 0s - loss: 21.6556 - val_loss: 84.0935
    Epoch 47/100
     - 0s - loss: 21.5802 - val_loss: 85.0446
    Epoch 48/100
     - 0s - loss: 20.8020 - val_loss: 99.9986
    Epoch 49/100
     - 0s - loss: 22.7749 - val_loss: 86.3418
    Epoch 50/100
     - 0s - loss: 21.4987 - val_loss: 82.9141
    Epoch 51/100
     - 0s - loss: 20.1509 - val_loss: 70.0507
    Epoch 52/100
     - 0s - loss: 19.7685 - val_loss: 78.3498
    Epoch 53/100
     - 0s - loss: 19.3978 - val_loss: 91.0906
    Epoch 54/100
     - 0s - loss: 19.4627 - val_loss: 89.7235
    Epoch 55/100
     - 0s - loss: 19.9207 - val_loss: 83.2194
    Epoch 56/100
     - 0s - loss: 17.1787 - val_loss: 86.6503
    Epoch 57/100
     - 0s - loss: 17.3483 - val_loss: 80.8701
    Epoch 58/100
     - 0s - loss: 16.9797 - val_loss: 86.4298
    Epoch 59/100
     - 0s - loss: 17.3220 - val_loss: 71.7838
    Epoch 60/100
     - 0s - loss: 18.1543 - val_loss: 93.6724
    Epoch 61/100
     - 0s - loss: 17.3914 - val_loss: 93.8378
    Epoch 62/100
     - 0s - loss: 18.0779 - val_loss: 79.7002
    Epoch 63/100
     - 0s - loss: 16.3311 - val_loss: 77.5905
    Epoch 64/100
     - 0s - loss: 16.7341 - val_loss: 79.1023
    Epoch 65/100
     - 0s - loss: 17.1575 - val_loss: 85.5779
    Epoch 66/100
     - 0s - loss: 17.9837 - val_loss: 74.6997
    Epoch 67/100
     - 0s - loss: 17.2643 - val_loss: 82.0907
    Epoch 68/100
     - 1s - loss: 15.8843 - val_loss: 75.8042
    Epoch 69/100
     - 0s - loss: 15.4484 - val_loss: 83.0589
    Epoch 70/100
     - 0s - loss: 16.7637 - val_loss: 86.8050
    Epoch 71/100
     - 0s - loss: 15.5778 - val_loss: 79.2986
    Epoch 72/100
     - 0s - loss: 14.6776 - val_loss: 84.1683
    Epoch 73/100
     - 0s - loss: 15.5128 - val_loss: 73.5329
    Epoch 74/100
     - 0s - loss: 15.6201 - val_loss: 75.8598
    Epoch 75/100
     - 0s - loss: 16.1412 - val_loss: 85.5103
    Epoch 76/100
     - 0s - loss: 15.3371 - val_loss: 76.8360
    Epoch 77/100
     - 1s - loss: 14.8967 - val_loss: 77.7843
    Epoch 78/100
     - 0s - loss: 16.8250 - val_loss: 74.1633
    Epoch 79/100
     - 0s - loss: 14.5558 - val_loss: 77.7522
    Epoch 80/100
     - 0s - loss: 14.3947 - val_loss: 78.0035
    Epoch 81/100
     - 0s - loss: 17.1305 - val_loss: 100.7723
    Epoch 82/100
     - 0s - loss: 15.0175 - val_loss: 93.5878
    Epoch 83/100
     - 0s - loss: 16.4044 - val_loss: 82.4757
    Epoch 84/100
     - 1s - loss: 14.6794 - val_loss: 73.8707
    Epoch 85/100
     - 0s - loss: 13.9855 - val_loss: 80.1784
    Epoch 86/100
     - 0s - loss: 14.5158 - val_loss: 73.6610
    Epoch 87/100
     - 1s - loss: 17.7804 - val_loss: 75.9934
    Epoch 88/100
     - 0s - loss: 15.0285 - val_loss: 82.2965
    Epoch 89/100
     - 0s - loss: 14.3080 - val_loss: 82.6318
    Epoch 90/100
     - 0s - loss: 14.1243 - val_loss: 80.0523
    Epoch 91/100
     - 0s - loss: 14.5383 - val_loss: 78.4820
    Epoch 92/100
     - 0s - loss: 14.8898 - val_loss: 84.6617
    Epoch 93/100
     - 0s - loss: 13.8225 - val_loss: 85.9752
    Epoch 94/100
     - 0s - loss: 13.8833 - val_loss: 94.7483
    Epoch 95/100
     - 0s - loss: 13.9076 - val_loss: 74.2256
    Epoch 96/100
     - 0s - loss: 14.1433 - val_loss: 84.9972
    Epoch 97/100
     - 0s - loss: 15.0815 - val_loss: 95.7865
    Epoch 98/100
     - 0s - loss: 15.4445 - val_loss: 76.4110
    Epoch 99/100
     - 0s - loss: 13.6041 - val_loss: 81.5492
    Epoch 100/100
     - 0s - loss: 13.4281 - val_loss: 94.4645





    <keras.callbacks.History at 0x7fa49469b470>




```python
model.predict(predictors_norm)[0:10]
```




    array([[74.447426],
           [75.09273 ],
           [40.11295 ],
           [42.380333],
           [45.018127],
           [48.758713],
           [43.85791 ],
           [37.155243],
           [44.153145],
           [41.2229  ]], dtype=float32)



### Thank you for completing this lab!

This notebook was created by [Alex Aklson](https://www.linkedin.com/in/aklson/). I hope you found this lab interesting and educational. Feel free to contact me if you have any questions!

This notebook is part of a course on **Coursera** called *Introduction to Deep Learning & Neural Networks with Keras*. If you accessed this notebook outside the course, you can take this course online by clicking [here](https://cocl.us/DL0101EN_Coursera_Week3_LAB1).

<hr>

Copyright &copy; 2019 [IBM Developer Skills Network](https://cognitiveclass.ai/?utm_source=bducopyrightlink&utm_medium=dswb&utm_campaign=bdu). This notebook and its source code are released under the terms of the [MIT License](https://bigdatauniversity.com/mit-license/).
