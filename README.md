# 2017-tfm-nuria-oyaga
Below are explained the different steps that are taken in the realization of my TFM. The information displayed is sorted from most recent to oldest so that the latest updates can be accessed more quickly and easily.

## New linear motion - Adding a degree of freedom
Since we have obtained a good result with the frames that include a linear motion we proceed to increase a degree of freedom in this type of data.

### Recurrent Neural Networks
I used the same structure that in the previous training:
<p align="center">
  <img width="550" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/linear_point_255_y0/15_False_relu_categorical_crossentropy_10_properties.png">
</p>
In this case we obtain a result that we could consider as expected, these networks are able to better capture the temporal relationship and obtain a better performance than in the previous case.
<p align="center">
  <img width="450" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/linear_point_255_y0/15_False_relu_categorical_crossentropy_10_history.png">
</p>
<p align="center">
  <img width="350" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/linear_point_255_y0/15_False_relu_categorical_crossentropy_10_error_hist.png">
</p>
<p align="center">
  <img width="600" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/linear_point_255_y0/15_False_relu_categorical_crossentropy_10_rel_error_hist.png">
</p>
In the next image you can see target frame in the samples where the errors (absolute and relative) are maximum. As I mentioned, this type of structure has improved the results and the error made, despite remaining high, is reduced compared to the previous structure.
<p align="center">
  <img width="650" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/linear_point_255_y0/15_False_relu_categorical_crossentropy_10_max_error.png">
</p>
In order to improve the obtained results we will choose to increase the number of samples in consequence with the complexity of the problem and modify the structure of the network in the same way.

### Non Recurrent Neural Networks 
I used the same structure that in the previous training:
<p align="center">
  <img width="350" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_linear/5000_samples_y0/15_False_relu_categorical_crossentropy_10_properties.png">
</p>
As I mentioned earlier, with this new degree of freedom the complexity of the problem increases and, as expected, the performance of the network is not good.
<p align="center">
  <img width="450" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_linear/5000_samples_y0/15_False_relu_categorical_crossentropy_10_history.png">
</p>
<p align="center">
  <img width="350" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_linear/5000_samples_y0/15_False_relu_categorical_crossentropy_10_error_hist.png">
</p>
<p align="center">
  <img width="600" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_linear/5000_samples_y0/15_False_relu_categorical_crossentropy_10_rel_error_hist.png">
</p>
In the next image you can see target frame in the samples where the errors (absolute and relative) are maximum. Consequently, the error made is very high.
<p align="center">
  <img width="650" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_linear/5000_samples_y0/15_False_relu_categorical_crossentropy_10_max_error.png">
</p>

### New data
Recall that the function that governs this type of data is as follows:
```ruby
  g = lambda y, m: (m * y) + y0
```

In the previous examples we kept constant the value of y0 in the middle of the frame, that is:
```ruby
  y0 = int(height/2)
```

In this new case we will give the position y0 a random value within the range of the frame:
```ruby
  y0 = y0 = random.randint(0, height - 1)
```

With this new degree of freedom we increase the variety in the samples and complicate the problem to the neural network to predict the desired value. Below you can see an example of these new samples:

<p align="center">
  <a href="https://www.youtube.com/watch?v=z5rt7O1bHNw" target="_blank"><img src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Fronts/frames_linear_point_y0_sample.png" 
  alt="Frames linear point sample with random y0" width="500"/></a>
</p>

## Recurrent Neural Networks - New structure 
In view of the results obtained previously I have replaced the simple LSTM layer with a ConvLSTM which computes convolutional operations in both the input and the recurrent transformations.

### Parabolic motion
I used the same structure:
<p align="center">
  <img width="350" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/parabolic_point_255/ConvLSTM/15_False_relu_categorical_crossentropy_10_properties.png">
</p>
As in the previous case the performance of the network is improved and the results are practically the same as in the non-recurrent.
<p align="center">
  <img width="450" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/parabolic_point_255/ConvLSTM/15_False_relu_categorical_crossentropy_10_history.png">
</p>
<p align="center">
  <img width="350" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/parabolic_point_255/ConvLSTM/15_False_relu_categorical_crossentropy_10_error_hist.png">
</p>
<p align="center">
  <img width="600" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/parabolic_point_255/ConvLSTM/15_False_relu_categorical_crossentropy_10_rel_error_hist.png">
</p>
In the next image you can see target frame in the samples where the errors (absolute and relative) are maximum. Although the maximum error is greater than in the non-recurrent case, in terms of the average a very similar result is obtained.
<p align="center">
  <img width="650" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/parabolic_point_255/ConvLSTM/15_False_relu_categorical_crossentropy_10_max_error.png">
</p>

### Linear motion
The new structure is as follows:
<p align="center">
  <img width="350" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/linear_point_255/ConvLSTM/15_False_relu_categorical_crossentropy_10_properties.png">
</p>
With the new structure the performance of the network is improved and the results are practically the same as in the non-recurrent case that were pretty good
<p align="center">
  <img width="450" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/linear_point_255/ConvLSTM/15_False_relu_categorical_crossentropy_10_history.png">
</p>
<p align="center">
  <img width="350" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/linear_point_255/ConvLSTM/15_False_relu_categorical_crossentropy_10_error_hist.png">
</p>
<p align="center">
  <img width="600" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/linear_point_255/ConvLSTM/15_False_relu_categorical_crossentropy_10_rel_error_hist.png">
</p>
In the next image you can see target frame in the samples where the errors (absolute and relative) are maximum. The maximum error committed coincides with the non-recurrent case and in terms of the average a very similar result is obtained.
<p align="center">
  <img width="650" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/linear_point_255/ConvLSTM/15_False_relu_categorical_crossentropy_10_max_error.png">
</p>

## Recurrent Neural Networks - Parabolic and linear data
The performance obtained with the previous networks is still improved, so we chose to include the recurrent networks to try to improve it.

### Parabolic motion
Although it seems clear that the structure of the network is not adequate to the problem, I have trained the same network with the samples of the parabolic motion:
<p align="center">
  <img width="550" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/parabolic_point_255/LSTM/15_False_relu_categorical_crossentropy_10_properties.png">
</p>
Indeed, the results obtained are just as bad as in the previous case.
<p align="center">
  <img width="450" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/parabolic_point_255/LSTM/15_False_relu_categorical_crossentropy_10_history.png">
</p>
<p align="center">
  <img width="350" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/parabolic_point_255/LSTM/15_False_relu_categorical_crossentropy_10_error_hist.png">
</p>
<p align="center">
  <img width="600" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/parabolic_point_255/LSTM/15_False_relu_categorical_crossentropy_10_rel_error_hist.png">
</p>
In the next image you can see target frame in the samples where the errors (absolute and relative) are maximum. Consequently with the above mentioned the error are too high.
<p align="center">
  <img width="650" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/parabolic_point_255/LSTM/15_False_relu_categorical_crossentropy_10_max_error.png">
</p>
In view of the results obtained in this section, we continue the research modifying the structure of the recurrent network to improve performance.

### Linear motion
I have used the following structure:
<p align="center">
  <img width="550" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/linear_point_255/LSTM/15_False_relu_categorical_crossentropy_10_properties.png">
</p>
With the proposed structure the performance is not good and the error is very high.
<p align="center">
  <img width="450" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/linear_point_255/LSTM/15_False_relu_categorical_crossentropy_10_history.png">
</p>
<p align="center">
  <img width="350" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/linear_point_255/LSTM/15_False_relu_categorical_crossentropy_10_error_hist.png">
</p>
<p align="center">
  <img width="600" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/linear_point_255/LSTM/15_False_relu_categorical_crossentropy_10_rel_error_hist.png">
</p>
In the next image you can see target frame in the samples where the errors (absolute and relative) are maximum. As I mentioned earlier, the error made with this network is excessively high.
<p align="center">
  <img width="650" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/linear_point_255/LSTM/15_False_relu_categorical_crossentropy_10_max_error.png">
</p>

## Non Recurrent Neural Networks - With more samples
As we saw in the previous training the first thing we must do to increase the performance is to increase the number of samples according to the complexity of the data.

### Parabolic motion
I used 10000 samples instead of 1000 and the same structure:
<p align="center">
  <img width="300" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_parabolic/10000_samples/15_False_relu_categorical_crossentropy_10_properties.png">
</p>
In the same way as in the case of linear movement, as the number of samples increases, the performance of the network improves and a greater stabilization of the same is achieved in the training. Despite this, the error continues to be improvable.
<p align="center">
  <img width="450" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_parabolic/10000_samples/15_False_relu_categorical_crossentropy_10_history.png">
</p>
<p align="center">
  <img width="350" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_parabolic/10000_samples/15_False_relu_categorical_crossentropy_10_error_hist.png">
</p>
<p align="center">
  <img width="600" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_parabolic/10000_samples/15_False_relu_categorical_crossentropy_10_rel_error_hist.png">
</p>
In the next image you can see target frame in the samples where the errors (absolute and relative) are maximum. As in the case with less samples, the error committed is not null, although the maximum error committed is quite small. The improvement is considerable with the increase in the number of samples, however, as I mentioned before it is still far from having a good performance.
<p align="center">
  <img width="650" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_parabolic/10000_samples/15_False_relu_categorical_crossentropy_10_max_error.png">
</p>

### Linear motion
I used 5000 samples instead of 1000 and the same structure:
<p align="center">
  <img width="300" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_linear/5000_samples/15_False_relu_categorical_crossentropy_10_properties.png">
</p>
As the number of samples increases, the performance of the network improves and a greater stabilization of the same is achieved in the training.
<p align="center">
  <img width="450" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_linear/5000_samples/15_False_relu_categorical_crossentropy_10_history.png">
</p>
<p align="center">
  <img width="350" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_linear/5000_samples/15_False_relu_categorical_crossentropy_10_error_hist.png">
</p>
<p align="center">
  <img width="600" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_linear/5000_samples/15_False_relu_categorical_crossentropy_10_rel_error_hist.png">
</p>
In the next image you can see target frame in the samples where the errors (absolute and relative) are maximum. As in the case with less samples, the error committed is not null, although the maximum error committed is quite small. In addition, despite the fact that the maximum error is the same, the average error has decreased with increasing number of samples.
<p align="center">
  <img width="650" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_linear/5000_samples/15_False_relu_categorical_crossentropy_10_max_error.png">
</p>

## Non Recurrent Neural Networks - With new data types
First we do the training of non-recucurrent networks with the new types of data to check the scope they give us.

### Parabolic motion
I used the same 2D convolutional network structure:
<p align="center">
  <img width="300" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_parabolic/1000_samples/2_False_relu_categorical_crossentropy_10_properties.png">
</p>
In this case the movement is too complicated and the number of samples is very small so the network is not able to capture it, making it difficult to stabilize the network.
<p align="center">
  <img width="450" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_parabolic/1000_samples/2_False_relu_categorical_crossentropy_10_history.png">
</p>
<p align="center">
  <img width="600" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_parabolic/1000_samples/2_False_relu_categorical_crossentropy_10_error_hist.png">
</p>
In the next image you can see target frame in the samples where the errors (absolute and relative) are maximum. The mistake made is very large and the predicted position is not close to the real one.
<p align="center">
  <img width="800" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_parabolic/1000_samples/2_False_relu_categorical_crossentropy_10_max_error.png">
</p>

Due to the problems that we find in these new networks, I will carry out a new training with a greater number of samples and a different structure to improve the results.
  
### Linear motion
As in the URM case, I started training a 2D convolutional network whose structure can be seen in the following figure:
<p align="center">
  <img width="300" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_linear/1000_samples/2_False_relu_categorical_crossentropy_10_properties.png">
</p>
As in the two previous cases, the network manages to reduce and stabilize the loss function in only a few epochs but, because the sequences begin to get more complicated, this network is not able to capture 100% of the different movements. 
<p align="center">
  <img width="450" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_linear/1000_samples/2_False_relu_categorical_crossentropy_10_history.png">
</p>
<p align="center">
  <img width="600" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_linear/1000_samples/2_False_relu_categorical_crossentropy_10_error_hist.png">
</p>
In the next image you can see target frame in the samples where the errors (absolute and relative) are maximum. As we commented previously, the error committed is not null, although the maximum error committed is quite small.
<p align="center">
  <img width="800" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_linear/1000_samples/2_False_relu_categorical_crossentropy_10_max_error.png">
</p>
  
## New data types
After making a learning with simpler data and see that the performance is good we decided to complicate things to the network. For this I have established two new types of movements in space for the white point of the frames: linear movement and parabolic movement, which will combine with the URM movement in time.

The way to obtain these new frames is the same as before, we set the position in x by means of a URM movement but this time, instead of maintaining the height of the object (position y) constant, we will modify it according to a function.

The problem that we can find in this type of samples is that we are modifying the height, a value that must be integer, by means of a function that accepts decimal values, which causes a rounding to be necessary. Depending on the sample, this rounding can make the movement not seem as natural as it would be because it is possible that the height does not change from one instant to the next.

### Parabolic motion
The function for this motion type is:
```ruby
  g = lambda y, a, b: (a * (y ** 2)) + (b * y) + y0
```

And in the next you can see a sample of this:
<p align="center">
  <a href="https://www.youtube.com/watch?v=aC5IR28P5vg" target="_blank"><img src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Fronts/sample_0.png" 
  alt="Frames parabolic point sample" width="500"/></a>
</p>

### Linear motion
For this motion type the function is:
```ruby
  g = lambda y, m: (m * y) + y0
```

And in the next you can see a sample of this:
<p align="center">
  <a href="https://www.youtube.com/watch?v=6M4slvtwdr0" target="_blank"><img src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Fronts/sample_0.png" 
  alt="Frames linear point sample" width="500"/></a>
</p>

## Recurrent Neural Networks - LSTM
Once we have tested the prediction with simple networks without recurrences, we start to train, with the same data, networks that incorporate recurrence. In particular, we will focus the work on the LSTM networks helping us with the tutorials in https://machinelearningmastery.com/category/lstm/

### URM Point Frames dataset
For the same reason that in the case of vectors we decided to develop a recurrent network for this type of data, verifying that the implementation we made is correct.

In this case we must modify the structure of the data in the following way:
```ruby
  input_shape=(n_samples, know_points=20, height=80, width=120, channels=1)
  output_shape = height * width * channels
```
Due to the computational load involved in the training and evaluation of this network it is not yet possible to show the result of the evaluation, although its structure and evolution in training is shown.

<p align="center">
  <img width="500" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/URM_point_255/64_False_relu_categorical_crossentropy_10_properties.png">
</p>
<p align="center">
  <img width="450" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Frames/URM_point_255/64_False_relu_categorical_crossentropy_10_history.png">


### URM Vectors dataset
Although the result with non-recurrent networks was perfect, we decided to use this data to start with the new networks and verify that we have understood their implementation.

As in the non-recurrent case, the first thing we must do is to resize the data to ensure that the input shape is correct.nothing changes with respect to non-recurrent networks, so we must have the following shapes:

```ruby
  input_shape=(n_samples, know_points=20, vector_length=320)
  output_shape = vector_length
```
For this type of data we have trained a simple LSTM network whose structure can be seen in the following figure:
<p align="center">
  <img width="300" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Vector_URM/15_False_relu_categorical_crossentropy_10_properties.png">
</p>
As in the previous case, the network manages to reduce and stabilize the loss function in only a few epochs, without any error.
<p align="center">
  <img width="450" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Vector_URM/15_False_relu_categorical_crossentropy_10_history.png">
</p>
<p align="center">
  <img width="600" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Vector_URM/15_False_relu_categorical_crossentropy_10_error_hist.png">
</p>
In the next image you can see the samples where the errors (absolute and relative) are maximum. In this case we have not obtained any error so the first sample is shown.
<p align="center">
  <img width="600" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Recurrent/Vector_URM/15_False_relu_categorical_crossentropy_10_max_error.png">
</p>

### Linear Functions dataset
Due to the simplicity of this data and the good result obtained with non-recurrent networks, we have decided not to train recurrent networks with this type of data.

## Non-Recurrent Neural Networks
The first step was to try to solve the problem of prediction with a classical neural network that does not use recurrence. We carry out the training of different networks with the generated data.

### URM Point Frames dataset
As in the functions case, the first thing we must do to train the network for prediction is to resize the data to ensure that the input shape is correct.In this case we must have the following shapes:
```ruby
  input_shape=(n_samples, know_points=20, height=80, width=120)
  output_shape = height * width
```
For this type of data we have trained 2D convolutional network whose structure can be seen in the following figure:
<p align="center">
  <img width="300" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_URM/15_False_relu_categorical_crossentropy_10_properties.png">
</p>
As in the two previous cases, the network manages to reduce and stabilize the loss function in only a few epochs, without any error.
<p align="center">
  <img width="450" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_URM/15_False_relu_categorical_crossentropy_10_history.png">
</p>
<p align="center">
  <img width="600" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_URM/15_False_relu_categorical_crossentropy_10_error_hist.png">
</p>
In the next image you can see target frame in the samples where the errors (absolute and relative) are maximum. In this case we have not obtained any error so the first sample is shown.
<p align="center">
  <img width="800" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Frame_point_URM/15_False_relu_categorical_crossentropy_10_max_error.png">
</p>

### URM Vectors dataset
As in the functions case, the first thing we must do to train the network for prediction is to resize the data to ensure that the input shape is correct.In this case we must have the following shapes:

```ruby
  input_shape=(n_samples, know_points=20, vector_length=320)
  output_shape = vector_length
```
For this type of data we have trained 1D convolutional network whose structure can be seen in the following figure:
<p align="center">
  <img width="300" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Vector_URM/15_False_relu_categorical_crossentropy_10_properties.png">
</p>
Aas expected and as it happened in the non-recurring case, the network manages to reduce and stabilize the loss function in only a few epochs, without any error.
<p align="center">
  <img width="450" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Vector_URM/15_False_relu_categorical_crossentropy_10_history.png">
</p>
<p align="center">
  <img width="600" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Vector_URM/15_False_relu_categorical_crossentropy_10_error_hist.png">
</p>
In the next image you can see the samples where the errors (absolute and relative) are maximum. In this case we have not obtained any error so the first sample is shown.
<p align="center">
  <img width="600" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Vector_URM/15_False_relu_categorical_crossentropy_10_max_error.png">
</p>

### Linear Functions dataset
The first thing we must do to train the network for prediction is to resize the data to ensure that the input shape is correct. For the function data we must have the following shapes:

```ruby
  input_shape=(n_samples, know_points=20, 1)
  output_shape = 1
```

For this type of data we have trained a simple MLP whose structure can be seen in the following figure:
<p align="center">
  <img width="280" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Linear_function/15_False_relu_mean_squared_error_10_properties.png">
</p>
As you can see in the following image, the network manages to reduce and stabilize the loss function in only a few epochs, obtaining a very reduced error.
<p align="center">
  <img width="450" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Linear_function/15_False_relu_mean_squared_error_10_history.png">
</p>
<p align="center">
  <img width="600" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Linear_function/15_False_relu_mean_squared_error_10_error_hist.png">
</p>
In the next image you can see the samples where the errors (absolute and relative) are maximum. In the case of relative error, when the line has a small slope the relative error is very high, this is because we divide by a very small value.

<p align="center">
  <img width="400" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Models/Non-Recurrent/Linear_function/15_False_relu_mean_squared_error_10_max_error.png">
</p>

## Data types
We will handle data of different nature that increase the degree of difficulty. The code to generate these samples can be found in https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/tree/master/Generator

### URM Point Frames dataset
The next step to increase the complexity is to increase one more dimension. In this way, each sample will consist of 20 + 1 images (frames) in which the URM movement of an object will be represented through the time that is represented with a single pixel.

In the following video you can see an example of a sample of this type of samples:
<p align="center">
  <a href="http://www.youtube.com/watch?v=RCEWNrTaYi8" target="_blank"><img src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Fronts/sample_0.png" 
  alt="Frames URM point sample" width="500"/></a>
</p>

As in the previous case, the speed of the object is limited so that in the prediction the object is always in the image.

### URM Vectors dataset
We increase the complexity of the previous data by increasing a dimension. We have created several 1D images in which the position of an object is represented at each moment of time. Each sample consists of 20 + 1 vectors, so that each vector consists of 320 positions and only activates (has a value of 255) that corresponds to the position in which the object would be found. To calculate the position we use a URM movement formula.

In the following figure you can see an example of a sample of this type of samples:
<p align="center">
  <img width="580" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/vector_sample.png">
</p>
The 2D image represents in each row a continuous moment of time with the exception of the last row that corresponds to the position to be predicted with a gap of 10. In addition, the speed of the object is limited so that in the prediction the object is always in the image.

### Linear Functions dataset
It is the simplest data to handle. The input of the network is a sequence of 20 numbers that follow the function of a line and the value that the network will return to us is the corresponding value of the function at point 20 + gap.

The samples are stored in a .txt file in which each line corresponds to a sample and the values of the function are stored in points [0,19] and 19 + gap.

In the following figure you can see an example of a sample of this type of functions:
<p align="center">
  <img width="460" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/linear_function_sample.png">
</p>
To reduce complexity and avoid infinite slope lines, samples have been generated with a limitation in their slope defined in the configuration file.
