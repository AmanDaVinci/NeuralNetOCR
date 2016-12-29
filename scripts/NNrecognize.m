function prediction = NNrecognize(imageFile,...
	learntWeights = '../data/learntWeights', cropPercentage=0, rotStep=0)

%%	NNRecognize:
%				Recognizes input image using a trained neural network
%
%	Usage: NNrecognize('myDigit.jpg', 'learntWeights', 100, 1);
%
%
%	Output: Prediction as a string
%
%   First parameter: Image file name
%             Could be bigger than 20 x 20 px, it will
%             be resized to 20 x 20. Better if used with
%             square images but not required.
% 
%   Second parameter: cropPercentage (any number between 0 and 100)
%             0  0% will be cropped (optional, no needed for square images)
%            50  50% of available croping will be cropped
%           100  crop all the way to square image (for rectangular images)
% 
%   Third parameter: rotStep
%            -1  rotate image 90 degrees CCW
%             0  do not rotate (optional)
%             1  rotate image 90 degrees CW
%
