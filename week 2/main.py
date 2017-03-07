from numpy import exp, array, random, dot

class NeuralNetwork():
	def __init__(self):
		#seed the random number generator 
		random.seed(1)

		#A neuron has 3 inputs and 1 output
		#random weight to a 3 x 1 matrix, with values in teh range of -1 to 1
		#and mean 0
		self.synaptic_weights = 2 * random.random((3,1)) - 1

	#Sigmoid function which describes and s shaped curve
	#pass the weighted sum of the inputs thorugh this function
	#to normalise them between 0 and 1 (so a probability)
	def __sigmoid(self, x):
		return 1 /(1+ exp(-x))
	#gradient of our sigmoid curve
	def __sigmoid_derivative(self, x):
		return x * (1-x)

	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
		for iteration in xrange(number_of_training_iterations):
			#pass the iteration set
			output = self.predict(training_set_inputs)

			error = training_set_outputs - output

			adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
			self.synaptic_weights += adjustment
	def predict(self, inputs):
		#pass inputs through our neural network to a single neuron
		return self.__sigmoid(dot(inputs, self.synaptic_weights))

if __name__ == '__main__':
	neural_network = NeuralNetwork()

	print "Random starting synaptic weights:"
	print neural_network.synaptic_weights


	training_set_inputs = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
	training_set_outputs = array([[0,1,1,0]]).T

	#train the neural net using a training set
	#do it 10,000 times
	neural_network.train(training_set_inputs, training_set_outputs, 1000)

	print "New synaptic weights after training: "
	print neural_network.synaptic_weights

	print "Predicting"
	print neural_network.predict(array([1,0,0]))
