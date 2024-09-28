/**
 * @author RusFortunat, i.e., Ruslan Mukhamadiarov
 */
import java.util.concurrent.ThreadLocalRandom;

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
	// Defined here because we will use stochastic gradient descent optimization -- 
	// collect error from minibatches before backpropagating it
	private double[][] GRADfirstLayerWeights;
    private double[] GRADfirstLayerBiases;
    private double[][] GRADsecondLayerWeights;
    private double[] GRADsecondLayerBiases;
	
	// constructor; initialize a fully-connected neural network with random weights and biases
    public Network(double learningRate, int minibatchSize, int inputSize, int hiddenSize, int outputSize){
        this.learningRate = learningRate;
		this.minibatchSize = minibatchSize;
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.hiddenVector = new double[hiddenSize];
        this.outputVector = new double[outputSize];
        this.firstLayerWeights = new double[inputSize][hiddenSize];
        this.firstLayerBiases = new double[hiddenSize];
        this.secondLayerWeights = new double[hiddenSize][outputSize];
        this.secondLayerBiases = new double[outputSize];
		this.GRADfirstLayerWeights = new double[inputSize][hiddenSize];
		this.GRADfirstLayerBiases = new double[hiddenSize];
		this.GRADsecondLayerWeights = new double[hiddenSize][outputSize];
		this.GRADsecondLayerBiases = new double[outputSize];
		
        // it is a good practice to limit distribution to inverse vector size 
        double rangeW1 = 1.0/inputSize; 
        for(int j = 0; j < hiddenSize; j++){
            firstLayerBiases[j] = ThreadLocalRandom.current().nextDouble(-rangeW1,rangeW1);
            for(int i = 0; i < hiddenSize; i++) {
                firstLayerWeights[i][j] = ThreadLocalRandom.current().nextDouble(-rangeW1,rangeW1);
            }
        }
        double rangeW2 = 1.0/hiddenSize;
        for(int j = 0; j < outputSize; j++){
            secondLayerBiases[j] = ThreadLocalRandom.current().nextDouble(-rangeW2,rangeW2);
            for(int i = 0; i < hiddenSize; i++) {
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
        for(int i = 0; i < outputSize; i++){
            double sum = 0;
            for(int j = 0; j < hiddenSize; j++){
                double activation = secondLayerWeights[i][j]*input[j] + secondLayerBiases[i];
                if(activation > 0) sum+= activation; // ReLU activation
            }
            outputVector[i] = sum;
        }

        // SoftMax
        double sum = 0;
        for(Double element:outputVector) sum += Math.exp(element);
        for(Double element:outputVector) element = Math.exp(element) / sum;
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
	public void backpropagateError()
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
	
	
	public void train(Object trainImages, Object trainLabels, int minibatchSize){
		
		
	}
	
	public void test(Object testImages, Object testLabels){
		
		
	}
	
	public Object[] getNetworkParameteres(){
        return new Object[]{firstLayerWeights, firstLayerBiases, secondLayerWeights, secondLayerBiases};
    }
}
