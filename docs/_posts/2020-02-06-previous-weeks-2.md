---
title: "Previous weeks - Change in initial height"
excerpt: "Linear motion is further complicated by letting the point start at a random height."

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

## New data
Since we have obtained a good result with the frames that include a linear motion we proceed to increase a degree of freedom in this type of data letting the point start at a random height. In the following [link](https://roboticslaburjc.github.io/2017-tfm-nuria-oyaga/previous%20work/datasets/) you can find a description of this new type of images.

## Non Recurrent Neural Networks 
I used the same structure that in the previous training
{% include figure image_path="/assets/images/logbook/media/Models/Non-Recurrent/Frame_point_linear/5000_samples_y0/15_False_relu_categorical_crossentropy_10_properties.png" alt="2D Convolutional network structure" %}
As I mentioned earlier, with this new degree of freedom the complexity of the problem increases and, as expected, the performance of the network is not good.
{% include figure image_path="/assets/images/logbook/media/Models/Non-Recurrent/Frame_point_linear/5000_samples_y0/15_False_relu_categorical_crossentropy_10_history.png" alt="Loss history" %}
{% include figure image_path="/assets/images/logbook/media/Models/Non-Recurrent/Frame_point_linear/5000_samples_y0/15_False_relu_categorical_crossentropy_10_error_hist.png" alt="Error histogram" %}
{% include figure image_path="/assets/images/logbook/media/Models/Non-Recurrent/Frame_point_linear/5000_samples_y0/15_False_relu_categorical_crossentropy_10_rel_error_hist.png" alt="Relative error histogram" %}
In the next image you can see target frame in the samples where the errors (absolute and relative) are maximum. Consequently, the error made is very high.
{% include figure image_path="/assets/images/logbook/media/Models/Non-Recurrent/Frame_point_linear/5000_samples_y0/15_False_relu_categorical_crossentropy_10_max_error.png" alt="Relative and absolute error" %}

## Recurrent Neural Networks
I used the same structure that in the previous training:
{% include figure image_path="/assets/images/logbook/media/Models/Recurrent/Frames/linear_point_255_y0/15_False_relu_categorical_crossentropy_10_properties.png" alt="ConvLSTM network structure" %}
In this case we obtain a result that we could consider as expected, these networks are able to better capture the temporal relationship and obtain a better performance than in the previous case.
{% include figure image_path="/assets/images/logbook/media/Models/Recurrent/Frames/linear_point_255_y0/15_False_relu_categorical_crossentropy_10_history.png" alt="Loss history" %}
{% include figure image_path="/assets/images/logbook/media/Models/Recurrent/Frames/linear_point_255_y0/15_False_relu_categorical_crossentropy_10_error_hist.png" alt="Error histogram" %}
{% include figure image_path="/assets/images/logbook/media/Models/Recurrent/Frames/linear_point_255_y0/15_False_relu_categorical_crossentropy_10_rel_error_hist.png" alt="Relative error histogram" %}
In the next image you can see target frame in the samples where the errors (absolute and relative) are maximum. As I mentioned, this type of structure has improved the results and the error made, despite remaining high, is reduced compared to the previous structure.
{% include figure image_path="/assets/images/logbook/media/Models/Recurrent/Frames/linear_point_255_y0/15_False_relu_categorical_crossentropy_10_max_error.png" alt="Relative and absolute error" %}
In order to improve the obtained results we will choose to increase the number of samples in consequence with the complexity of the problem and modify the structure of the network in the same way.
