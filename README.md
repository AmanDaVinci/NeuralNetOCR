# NeuralNetOCR
A Neural Network trained for Optical Character Recognition 

### Handwritten Digit Recognition 

![Neural Network Prediction](results/NeuralNetPredictions.gif)

#### Training Set Accuracy: *97.52%*

### Pattern Digit Recognition 
![Neural Network Prediction](results/recognition.png)

### Vectorized Implementation of y Labels to Y matrix

I have implemented an interesting vectorized method to compute the multi-class representation of y matrix from the label vector y.

This algorithm works both in MATLAB and Octave. It is clear and expansive but easy to understand.
And this is m times faster, where m is the number of examples than the standard loop method.

	<script>
		recodeY(y, numExamples, numLabels):

		 Y_matrix = Generate a numExamples by numLabels matrix
		 Y_matrix = Unroll Y_matrix and Transpose it  
		  rowIndexes = Row Vector of numExamples elements having numLabels as each     element
		  columnIndexes = Consecutive Row Vector of  elements from 0 to numExample-1
		  indexes = rowIndexes * columnIndexes
		  y = Transpose of y + indexes
		  Y_matrix(y)  = 1
		  Y_matrix = reshape Y_matrix as numLabels by numExamples and transpose it
	</script>

In simple intuitive steps what we do is use y labels to index into the Y matrix and change those elements / features into 1.