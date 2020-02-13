---
title: "Previous weeks - Previous work"
excerpt: "All work done before resuming the thesis this year."

sidebar:
  nav: "docs"

classes: wide

categories:
- previous work

tags:
- logbook
- studying
- training

author: NuriaOF
pinned: false


---

## Training Non-Recurrent Neural Networks

The first step was to try to solve the problem of prediction with a classical neural network that does not use recurrence. We carry out the training of different networks with the generated data.

### Linear Functions dataset
The first thing we must do to train the network for prediction is to resize the data to ensure that the input shape is correct. For the function data we must have the following shapes:

```ruby
  input_shape=(n_samples, know_points=20, 1)
  output_shape = 1
```

For this type of data we have trained a simple MLP whose structure can be seen in the following figure:

{% include figure image_path="/assets/images/logbook/media/Models/Non-Recurrent/Linear_function/15_False_relu_mean_squared_error_10_properties.png" alt="MLP structure" %}

As you can see in the following image, the network manages to reduce and stabilize the loss function in only a few epochs, obtaining a very reduced error.

{% include figure image_path="/assets/images/logbook/media/Models/Non-Recurrent/Linear_function/15_False_relu_mean_squared_error_10_history.png" alt="Loss history" %}

{% include figure image_path="/assets/images/logbook/media/Models/Non-Recurrent/Linear_function/15_False_relu_mean_squared_error_10_error_hist.png" alt="Relative error histogram" %}

In the next image you can see the samples where the errors (absolute and relative) are maximum. In the case of relative error, when the line has a small slope the relative error is very high, this is because we divide by a very small value.

{% include figure image_path="/assets/images/logbook/media/Models/Non-Recurrent/Linear_function/15_False_relu_mean_squared_error_10_max_error.png" alt="Relative and absolute error" %}


### URM Vectors dataset
As in the functions case, the first thing we must do to train the network for prediction is to resize the data to ensure that the input shape is correct.In this case we must have the following shapes:

```ruby
  input_shape=(n_samples, know_points=20, vector_length=320)
  output_shape = vector_length
```

For this type of data we have trained 1D convolutional network whose structure can be seen in the following figure:

{% include figure image_path="/assets/images/logbook/media/Models/Non-Recurrent/Vector_URM/15_False_relu_categorical_crossentropy_10_properties.png" alt="1D Convolutional network structure" %}

As expected and as it happened in the non-recurring case, the network manages to reduce and stabilize the loss function in only a few epochs, without any error.

{% include figure image_path="/assets/images/logbook/media/Models/Non-Recurrent/Vector_URM/15_False_relu_categorical_crossentropy_10_history.png" alt="Loss history" %}

{% include figure image_path="/assets/images/logbook/media/Models/Non-Recurrent/Vector_URM/15_False_relu_categorical_crossentropy_10_error_hist.png" alt="Relative error histogram" %}

In the next image you can see the samples where the errors (absolute and relative) are maximum. In this case we have not obtained any error so the first sample is shown.

{% include figure image_path="/assets/images/logbook/media/Models/Non-Recurrent/Vector_URM/15_False_relu_categorical_crossentropy_10_max_error.png" alt="Relative and absolute error" %}

### URM Point Frames dataset
As in the functions case, the first thing we must do to train the network for prediction is to resize the data to ensure that the input shape is correct.In this case we must have the following shapes:

```ruby
  input_shape=(n_samples, know_points=20, height=80, width=120)
  output_shape = height * width
```

For this type of data we have trained 2D convolutional network whose structure can be seen in the following figure:

{% include figure image_path="/assets/images/logbook/media/Models/Non-Recurrent/Frame_point_URM/15_False_relu_categorical_crossentropy_10_properties.png" alt="2D Convolutional network structure" %}

As in the two previous cases, the network manages to reduce and stabilize the loss function in only a few epochs, without any error.

{% include figure image_path="/assets/images/logbook/media/Models/Non-Recurrent/Frame_point_URM/15_False_relu_categorical_crossentropy_10_history.png" alt="Loss history" %}

{% include figure image_path="/assets/images/logbook/media/Models/Non-Recurrent/Frame_point_URM/15_False_relu_categorical_crossentropy_10_error_hist.png" alt="Relative error histogram" %}

In the next image you can see target frame in the samples where the errors (absolute and relative) are maximum. In this case we have not obtained any error so the first sample is shown.

{% include figure image_path="/assets/images/logbook/media/Models/Non-Recurrent/Frame_point_URM/15_False_relu_categorical_crossentropy_10_max_error.png" alt="Relative and absolute error" %}
