# Handwritten-Character-Recognition-Using-Image-Processing
Handwritten characters are crucial in daily life, but recognizing them requires dealing with diverse styles and complex digitization processes. Deep learning approaches are used for offline recognition.

<p align="left">
At present, handwritten characters are increasingly used in daily life. Handwriting digits and character recognitions have become increasingly important in today's digitized world due to their practical applications in various day-to-day activities. The main challenge in handwritten character recognition is to deal with the enormous variety of handwriting styles by different writers. Systems that are used to recognize Handwriting letters, characters, and digits help people to solve more complex tasks that otherwise would be time-consuming and costly.
</p>

## Motivation

To implement an algorithm that will be able to perform image processing of HCR which is a process of automatic computer recognition of characters in optically scanned and digitized pages of text.

## Features

- Train a model to HCR.
- Change parameters and compare Loss vs. Accuracy
- Recognize handwritten characters.

## Method
### Flow Chart of the Process

<p align="center">
<img width="650" src="https://github.com/janithcbk123/Handwritten-Character-Recognition-Using-Image-Processing/blob/main/Supportive%20Images/Flowchart.jpg">
</p>

### Selecting Mode (Train or Test)

The System allows the use to Train a new model or use an existing trained model. When the training mode is selected, the system allows the user to select how many epochs should be used in the model or, the user can choose to “calculate the optimum number of epochs mode”. 

<p align="center">
  <img width="650" src="https://github.com/janithcbk123/Handwritten-Character-Recognition-Using-Image-Processing/blob/main/Supportive%20Images/Selecing%20Mode.jpg?raw=true">
</p>

Train Mode or Test mode can be selected by applying “1” or “0”.

<p align="center">
  <img width="350" src="https://github.com/janithcbk123/Handwritten-Character-Recognition-Using-Image-Processing/blob/main/Supportive%20Images/Train%20or%20Test.jpg?raw=true">
</p>

If the user selects “Test Mode” the System requests the user to set the number of test images in the “Drop_test_Image_Here” folder.

<p align="center">
  <img width="250" src="https://github.com/janithcbk123/Handwritten-Character-Recognition-Using-Image-Processing/blob/main/Supportive%20Images/Set%20Number%20of%20test%20Images%20-%20User%20Interface.jpg?raw=true">
</p>

If the user selects the “Train Mode” System, the user can calculate the optimum number of epochs or enter the pre-calculated value. 

<p align="center">
  <img width="400" src="https://github.com/janithcbk123/Handwritten-Character-Recognition-Using-Image-Processing/blob/main/Supportive%20Images/Set%20number%20of%20Epochs%20or%20Calculate%20optimum%20number%20of%20Epochs%20-%20User%20Interface.jpg?raw=true">
</p>



### Calculate the Optimum Number of epochs

<p align="left">
  The System allows the use to Train a new model or use an existing trained model. When the training mode is selected, the system allows the user to select how many epochs should be used in the model or, the user can choose to “calculate optimum number of epochs    mode”. 
</p>

<p align="left">
  An epoch is a term used in neural networks to indicate the number of passes of the entire training dataset the algorithm has completed. Datasets are usually grouped into batches. The number of epochs required depends on the size of the model and the variation     in the dataset. But, too many epochs may cause your model to over-fit the training data. It means that your model does not learn the data, it memorizes the data. You have to find the accuracy of validation data for each epoch or maybe iteration to investigate     whether it overfits or not. 
</p>

<p align="center">
  <img width="450" src="https://github.com/janithcbk123/Handwritten-Character-Recognition-Using-Image-Processing/blob/main/Supportive%20Images/Accuracy%20vs%20Number%20of%20Epochs.jpg?raw=true">
</p>

<p align="center">
  <img width="450" src="https://github.com/janithcbk123/Handwritten-Character-Recognition-Using-Image-Processing/blob/main/Supportive%20Images/Loss%20vs%20Number%20of%20Epochs.jpg?raw=true">
</p>



### Uploading Testing Images

<p align="left">
  The System allows the use of Test image recognition on custom images. To process multiple images at one time they need to be placed in the “Drop_Your_Image_Here” folder and numbered from “001”.
</p>

<p align="center">
  <img width="450" src="https://github.com/janithcbk123/Handwritten-Character-Recognition-Using-Image-Processing/blob/main/Supportive%20Images/Drop_Your_Images_Here%20folder%20and%20Numbering.jpg?raw=true">
</p>



## Results
<p align="center">
  <img align="center" alt="Coding" width="750" src="https://github.com/janithcbk123/Handwritten-Character-Recognition-Using-Image-Processing/blob/main/Supportive%20Images/Results/Accuracy%20vs%20Number%20of%20Epochs.jpg?raw=true">
</p>

<p align="center">
  <img width="350" src="https://github.com/janithcbk123/Handwritten-Character-Recognition-Using-Image-Processing/blob/main/Supportive%20Images/Results/Test%20Image%20'j'.jpg?raw=true">
  <img width="350" src="https://github.com/janithcbk123/Handwritten-Character-Recognition-Using-Image-Processing/blob/main/Supportive%20Images/Results/Test%20Image%20'A'.jpg?raw=true">
</p>

<p align="center">
  <img width="250" src="https://github.com/janithcbk123/Handwritten-Character-Recognition-Using-Image-Processing/blob/main/Supportive%20Images/Results/Test%20Image%20'j'.jpg?raw=true">
  <img width="250" src="https://github.com/janithcbk123/Handwritten-Character-Recognition-Using-Image-Processing/blob/main/Supportive%20Images/Results/Test%20Image%20'k'.jpg?raw=true">
  <img width="250" src="https://github.com/janithcbk123/Handwritten-Character-Recognition-Using-Image-Processing/blob/main/Supportive%20Images/Results/Test%20Image%20'K'%20(Cap).jpg?raw=true">
</p>



## Conclusion

<p align="left">
  When Epochs, we can see in the graph accuracy goes high. But the validation loss also goes high. So it is important to take the number of epochs that are high enough to be accurate but low enough to be less losses in validation. And also Epochs majorly affect   the Training time of the model. So it is easier to keep the Epochs number as minimal as possible.
</p>

<p align="left">
  In this case, we can see 23 epochs is a good value to be set. It has good accuracy, less losses in validation, and does not take so much time to train.
</p>


## Acknowledgements
I would like to acknowledge and give our thanks to Mr. B.G.D. Achintha Madhusanka and the DMX5314 module team for supporting the project.

