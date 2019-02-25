# 2017-tfm-nuria-oyaga
## Data types
For the purpose of this TFM we will handle data of different nature that increase the degree of difficulty. The code to generate these samples can be found in https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/tree/master/Generator
### Linear Functions dataset
It is the simplest data to handle. The input of the network is a sequence of 20 numbers that follow the function of a line and the value that the network will return to us is the corresponding value of the function at point 20 + gap.

The samples are stored in a .txt file in which each line corresponds to a sample and the values of the function are stored in points [0,19] and 19 + gap.

In the following figure you can see an example of a sample of this type of functions:
<p align="center">
  <img width="460" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/linear_function_sample.png">
</p>
To reduce complexity and avoid infinite slope lines, samples have been generated with a limitation in their slope defined in the configuration file.

### URM Vectors dataset
We increase the complexity of the previous data by increasing a dimension. We have created several 1D images in which the position of an object is represented at each moment of time. Each sample consists of 20 + 1 vectors, so that each vector consists of 320 positions and only activates (has a value of 255) that corresponds to the position in which the object would be found. To calculate the position we use a URM movement formula.

In the following figure you can see an example of a sample of this type of samples:
<p align="center">
  <img width="580" src="https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/vector_sample.png">
</p>
The 2D image represents in each row a continuous moment of time with the exception of the last row that corresponds to the position to be predicted with a gap of 10. In addition, the speed of the object is limited so that in the prediction the object is always in the image.

### URM Point Frames dataset
The next step to increase the complexity is to increase one more dimension. In this way, each sample will consist of 20 + 1 images (frames) in which the URM movement of an object will be represented through the time that is represented with a single pixel.

In the following video you can see an example of a sample of this type of samples:

[![URM point frame]("https://github.com/RoboticsURJC-students/2017-tfm-nuria-oyaga/blob/master/docs/Frames%20samples/White%20point/sample400_0.png")](https://www.youtube.com/watch?v=RCEWNrTaYi8)
