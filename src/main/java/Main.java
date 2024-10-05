/**
 * @author RusFortunat, i.e., Ruslan Mukhamadiarov
 */
 
public class Main {

	public static void main(String[] args){
		
		// Neural Network parameters 
		double learningRate = 0.001; // good practice to keep it small, between 0.001-0.0001
		int inputSize = 28*28; // images sizes
		int hiddenSize = 128;
		int outputSize = 10; // number of possible labels
		int batchSize = 100; // we will split the whole data set to minibatches for efficient training
		int trainingEpisodes = 10; // select how long you want to train your neural network
		
		Network imageClassifierNet = new Network(learningRate, inputSize, hiddenSize, outputSize);
		
		int trainDatasetSize = 60000;
		String trainImagesPath = "D:/Work/data_sets/MNIST_handwritten_digits/train-images.idx3-ubyte";
		String trainLabelsPath = "D:/Work/data_sets/MNIST_handwritten_digits/train-labels.idx1-ubyte";
		ReadData trainDataMNIST = new ReadData(trainImagesPath, trainLabelsPath, trainDatasetSize, inputSize);
		try{
			trainDataMNIST.loadImageData();
		}catch(Exception e){
			System.out.println("Error message: " + e.getMessage());
		}
		
		imageClassifierNet.train(trainDataMNIST, trainingEpisodes, batchSize);
		
		int testDatasetSize = 10000;
		String testImagesPath = "D:/Work/data_sets/MNIST_handwritten_digits/t10k-images.idx3-ubyte";
		String testLabelsPath = "D:/Work/data_sets/MNIST_handwritten_digits/t10k-labels.idx1-ubyte";
		ReadData testDataMNIST = new ReadData(testImagesPath, testLabelsPath, testDatasetSize, inputSize);
		try{
			testDataMNIST.loadImageData();
		}catch(Exception e){
			System.out.println("Error message: " + e.getMessage());
		}
                
		imageClassifierNet.test(testDataMNIST);
	}
 
}
