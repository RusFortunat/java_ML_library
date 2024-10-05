/**
 * @author RusFortunat, i.e., Ruslan Mukhamadiarov
 */
import java.util.concurrent.ThreadLocalRandom;
import java.util.Random;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.OptionalInt;

// at the moment just 1 hidden layer, generalization will come later
public class Network {
    private double learningRate;
    private int inputSize;
    private int hiddenSize;
    private int outputSize;
    private double[] hiddenVector; 
    private double[] outputVector;
    // NN params
    private double[][] firstLayerWeights;
    private double[] firstLayerBiases;
    private double[][] secondLayerWeights;
    private double[] secondLayerBiases;
    // it's ugly to define helping arrays here, but i'm not sure it will look better if we will be passing them around 
    private double[][] GRADfirstLayerWeights;
    private double[] GRADfirstLayerBiases;
    private double[][] GRADsecondLayerWeights;
    private double[] GRADsecondLayerBiases;
	
	// constructor; initialize a fully-connected neural network with random weights and biases
    public Network(double learningRate, int inputSize, int hiddenSize, int outputSize){
        this.learningRate = learningRate;
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.hiddenVector = new double[hiddenSize];
        this.outputVector = new double[outputSize];
        this.firstLayerWeights = new double[hiddenSize][inputSize];
        this.firstLayerBiases = new double[hiddenSize];
        this.secondLayerWeights = new double[outputSize][hiddenSize];
        this.secondLayerBiases = new double[outputSize];
        this.GRADfirstLayerWeights = new double[hiddenSize][inputSize];
        this.GRADfirstLayerBiases = new double[hiddenSize];
        this.GRADsecondLayerWeights = new double[outputSize][hiddenSize];
        this.GRADsecondLayerBiases = new double[outputSize];
		
        // it is a good practice to limit distribution to inverse vector size 
        double rangeW1 = 1.0/inputSize; 
        for(int i = 0; i < hiddenSize; i++){
            firstLayerBiases[i] = ThreadLocalRandom.current().nextDouble(-rangeW1,rangeW1);
            for(int j = 0; j < hiddenSize; j++) {
                firstLayerWeights[i][j] = ThreadLocalRandom.current().nextDouble(-rangeW1,rangeW1);
            }
        }
        double rangeW2 = 1.0/hiddenSize;
        for(int i = 0; i < outputSize; i++){
            secondLayerBiases[i] = ThreadLocalRandom.current().nextDouble(-rangeW2,rangeW2);
            for(int j = 0; j < hiddenSize; j++) {
                secondLayerWeights[i][j] = ThreadLocalRandom.current().nextDouble(-rangeW2,rangeW2);
            }
        }
    }
    
	
    // implements feed-forward propagation with ReLU activation
    public void forward(double[] input){
        // compute hidden activation values
        for(int i = 0; i < hiddenSize; i++){
            double sum = 0;
            for(int j = 0; j < inputSize; j++){
                double activation = firstLayerWeights[i][j]*input[j] + firstLayerBiases[i];
                if(activation > 0) sum+= activation; // ReLU activation
            }
            hiddenVector[i] = sum;
        }
        // compute output activations
        double totalSum = 0;
        for(int i = 0; i < outputSize; i++){
            double sum = 0;
            for(int j = 0; j < hiddenSize; j++){
                double activation = secondLayerWeights[i][j]*input[j] + secondLayerBiases[i];
                if(activation > 0) sum+= activation; // ReLU activation
            }
            outputVector[i] = sum;
			totalSum += Math.exp(sum); 
        }

        // SoftMax
        for(Double element:outputVector) element = Math.exp(element) / totalSum;
    }
	
	
    // Backpropagation; The math behind it is nicely explained in 3b1b youtube video on this topic
    public void computeGradients(double[] inputVector, double[] targetVector, int batchSize){
        // Second(final) layer W2 & B2 gradients
        for(int i=0; i < outputSize; i++){
            if(outputVector[i] > 0){ // comes from ReLU's derivative
                for(int j = 0; j < hiddenSize; j++){
                    if(hiddenVector[j] != 0){
                        GRADsecondLayerWeights[i][j] += (1.0/batchSize) * 2 * (outputVector[i] - targetVector[i]) * hiddenVector[j];
                    }
                }
                GRADsecondLayerBiases[i] += (1.0/batchSize) * 2 * (outputVector[i] - targetVector[i]);
            }
        }
        // First(starting) layer W1 & B1 gradients
        for(int i = 0; i < hiddenSize; i++){
            if(hiddenVector[i] > 0){ // comes from ReLU's derivative
                double secondLayerGrad = 0;
                for(int k = 0; k < outputSize; k++){
                    if(outputVector[k] > 0){ // comes from ReLU's derivative
                        secondLayerGrad += 2.0 * (outputVector[k] - targetVector[k]) * secondLayerWeights[k][i];
                    }
                }
                for(int j = 0; j < inputSize; j++){
                    if(inputVector[j] != 0){
                        GRADfirstLayerWeights[i][j] += (1.0/batchSize) * secondLayerGrad * inputVector[j];
                    }
                }
                GRADfirstLayerBiases[i] += (1.0/batchSize) * secondLayerGrad;
            }
        }
	}
	
	
    // update neural network parameters, i.e., optimizer 
    public void backpropagateError(){
        // first layer W1 & B1
        for(int i = 0; i < hiddenSize; i++){
            for(int j = 0; j < inputSize; j++){
                firstLayerWeights[i][j] -= learningRate * GRADfirstLayerWeights[i][j];
            }
            firstLayerBiases[i] -= learningRate * GRADfirstLayerBiases[i];
        }
        // second layer W2 & W2
        for(int i = 0; i < outputSize; i++){
            for(int j = 0; j < hiddenSize; j++){
                secondLayerWeights[i][j] -= learningRate * GRADsecondLayerWeights[i][j];
            }
            secondLayerBiases[i] -= learningRate * GRADsecondLayerBiases[i];
        }
    }	
	
	
    public void train(ReadData trainDataMNIST, int trainingEpisodes, int batchSize){
        double[][] trainImages = trainDataMNIST.getImages();
        int[] trainLabels = trainDataMNIST.getLabels();
        for(int episode = 0; episode < trainingEpisodes; episode++){
            double loss = 0; // should decrease

            // we will use this shuffle the dataset to have different minibatches each training episode
            long seed = System.nanoTime();
            ArrayList<Integer> indexList = new ArrayList<>();
            for(int i = 0; i < trainImages.length; i++) indexList.add(i);
            Collections.shuffle(indexList, new Random(seed));

            for(int minibatch = 0; minibatch < trainImages.length / batchSize; minibatch++){
				clearGradients();
				for(int i = 0; i < batchSize; i++){
					int trainImageIndex = indexList.get(i);
					double[] inputVector = trainImages[trainImageIndex];
					double[] targetVector = new double[outputSize]; // create target vector
					int label = trainLabels[trainImageIndex];
					targetVector[label] = 1.0;

					forward(inputVector); // get output vectors with probability distribution 
					computeGradients(inputVector, targetVector, batchSize); // compare target and output vectors
					for(int k = 0; k < outputSize; k++) loss += (1.0/trainImages.length)*Math.pow(outputVector[k] - targetVector[k],2);
				}
				backpropagateError(); // backpropagate accumulated error; optimize.step() in PyTorch
            }
            System.out.println("Episode: " + episode + ", loss = " + loss);
        }
        System.out.println("The training of neural network is done.");
    }
	
    public void test(ReadData testDataMNIST){
        double[][] testImages = testDataMNIST.getImages();
        int[] testLabels = testDataMNIST.getLabels();

        int correct = 0;
        for(int image = 0; image < testImages.length; image++){
            double[] inputVector = testImages[image];
            int target = testLabels[image];

            forward(inputVector);
            int maxLabel = getMaxIndex();
            //int maxValue = Arrays.stream(outputVector).max().getAsInt(); // WRONG!! YOU NEED INDEX
            if(maxLabel == target) correct++; 
        }
        System.out.println("Fraction of correctly predicted images: " + (1.0*correct/testImages.length));
    }
	
    public Object[] getNetworkParameteres(){
        return new Object[]{firstLayerWeights, firstLayerBiases, secondLayerWeights, secondLayerBiases};
    }
	
    public int getMaxIndex(){
        int maxIndex = 0;
        double maxValue = 0;
        for(int i = 0; i < outputVector.length; i++){
            if(outputVector[i] > maxValue){
                maxValue = outputVector[i];
                maxIndex = i;        
            }
        }
        return maxIndex;
    }
    
    public void clearGradients(){
        this.GRADfirstLayerWeights = null;
        this.GRADfirstLayerBiases = null;
        this.GRADsecondLayerWeights = null;
        this.GRADsecondLayerBiases = null;
        this.GRADfirstLayerWeights = new double[hiddenSize][inputSize];
        this.GRADfirstLayerBiases = new double[hiddenSize];
        this.GRADsecondLayerWeights = new double[outputSize][hiddenSize];
        this.GRADsecondLayerBiases = new double[outputSize];
    }
}
