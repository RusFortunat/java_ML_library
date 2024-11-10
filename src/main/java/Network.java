/**
 * @author RusFortunat, i.e., Ruslan Mukhamadiarov
 */
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.concurrent.ThreadLocalRandom;
import java.util.Random;
import java.util.ArrayList;
import java.util.Collections;

// at the moment just 1 hidden layer, generalization will come later
public class Network {
    private double learningRate;
    private int inputSize;
    private int hiddenSize;
    private int outputSize;
    private double predictionAccuracy;
    private double[] hiddenVector; 
    private double[] outputVector;
    // NN params
    private double[][] firstLayerWeights;
    private double[] firstLayerBiases;
    private double[][] secondLayerWeights;
    private double[] secondLayerBiases;
    // ugly, but i'm not sure it will look better if we will be passing them from method to method 
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
            for(int j = 0; j < inputSize; j++) {
                firstLayerWeights[i][j] = ThreadLocalRandom.current().nextDouble(-rangeW1,rangeW1);
            }
        }
        double rangeW2 = 1.0/hiddenSize;
        for(int i = 0; i < outputSize; i++){
            for(int j = 0; j < hiddenSize; j++) {
                secondLayerWeights[i][j] = ThreadLocalRandom.current().nextDouble(-rangeW2,rangeW2);
            }
        }
    }
	
    // implements feed-forward propagation with ReLU activation
    public void forward(double[] input){
        // forwardprop is very simple really; just do the following:
        // 1. Compute [z] = [firstLayerWeights][input] + [firstLayerBiases];
        // 2. Obtain the activation values of the hidden vector [y] by applying to [z] some activation function f([z]). 
        //    Here we use ReLU activation: f(z) > 0 ? z : 0;
        // 3. Repeat for the next layer with secondLayerWeights and secondLayerBiases to get the [output] vector.
        
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
        double totalSum = 0.0;
        for(int i = 0; i < outputSize; i++){
            double sum = 0;
            for(int j = 0; j < hiddenSize; j++){
                double activation = secondLayerWeights[i][j]*hiddenVector[j] + secondLayerBiases[i];
                if(activation > 0) sum+= activation; // ReLU activation
            }
            outputVector[i] = sum;
            totalSum += Math.exp(sum); 
        }

        // SoftMax
        for(int i = 0; i < outputSize; i++){ 
            outputVector[i] = Math.exp(outputVector[i]) / totalSum;
        }
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
	
    public void train(MNISTImageData trainDataMNIST, int trainingEpisodes, int batchSize){
        double[][] trainImages = trainDataMNIST.getImages();
        double datasetSize = trainImages.length;
        assert datasetSize != 0;
        int[] trainLabels = trainDataMNIST.getLabels();
        for(int episode = 0; episode < trainingEpisodes; episode++){
            double loss = 0; // should decrease as training goes on

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
                    for(int k = 0; k < outputSize; k++) loss += (1.0/datasetSize)*Math.pow(outputVector[k] - targetVector[k],2); // compute loss
                }
                backpropagateError(); // backpropagate accumulated error; optimize.step() in PyTorch
            }
            
            System.out.println("Episode: " + (episode+1) + ", loss = " + loss);
        }
        System.out.println("The training of neural network is done.");
    }
	
    public void test(MNISTImageData testDataMNIST){
        double[][] testImages = testDataMNIST.getImages();
        int[] testLabels = testDataMNIST.getLabels();

        int correct = 0;
        for(int image = 0; image < testImages.length; image++){
            double[] inputVector = testImages[image];
            int target = testLabels[image];

            forward(inputVector);
            int maxLabel = getMaxIndex();
            if(maxLabel == target) correct++; 
        }
        predictionAccuracy = 1.0*correct/testImages.length;
        System.out.println("Fraction of correctly predicted images: " + predictionAccuracy);
    }
	
    public int getMaxIndex(){
        int maxIndex = 0;
        double maxValue = outputVector[0];
        for(int i = 1; i < outputVector.length; i++){
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


    // save network parameters to a file, in a coma-separated fashion
    public void saveNetworkParameters(String filename){

        try{
            FileWriter file = new FileWriter(filename);

            // metadata
            file.write("Input layer size: " + inputSize
                    + "; Hidden layer size: " + hiddenSize + "; Output layer size: " + outputSize + "\n");
            file.write("Prediction accuracy: " + predictionAccuracy + "\n\n");

            file.write("First layer weights:\n");
            for(int i = 0; i < hiddenSize; i++){
                for(int j = 0; j < hiddenSize; j++) {
                    file.write(firstLayerWeights[i][j] + ",");
                }
            }
            file.write("\n\n");

            file.write("First layer biases:\n");
            for(int i = 0; i < hiddenSize; i++){
                file.write(firstLayerBiases[i] + ",");
            }
            file.write("\n\n");

            file.write("Second layer weights:\n");
            for(int i = 0; i < outputSize; i++){
                for(int j = 0; j < hiddenSize; j++) {
                    file.write(secondLayerWeights[i][j] + ",");
                }
            }
            file.write("\n\n");

            file.write("Second layer biases:\n");
            for(int i = 0; i < outputSize; i++){
                file.write(secondLayerBiases[i] + ",");
            }

            file.close();

        }catch(Exception e){
            System.out.println(e.getMessage());
        }
    }


    // printers -- were used in the debug phase

    public void printNetworkParameteres(){
        System.out.println("firstLayerWeights:");
        for(int i = 0; i < hiddenSize; i++){
            for(int j = 0; j < hiddenSize; j++) {
                System.out.print(firstLayerWeights[i][j] + " ");
            }
            System.out.println("");
        }
        System.out.println("\nfirstLayerBiases:");
        for(int i = 0; i < hiddenSize; i++){
            System.out.print(firstLayerBiases[i] + " ");
        }
        System.out.println("");
        
        System.out.println("\nsecondLayerWeights:");
        for(int i = 0; i < outputSize; i++){
            for(int j = 0; j < hiddenSize; j++) {
                System.out.print(secondLayerWeights[i][j] + " ");
            }
            System.out.println("");
        }
        System.out.println("\nsecondLayerBiases:");
        for(int i = 0; i < outputSize; i++){
            System.out.print(secondLayerBiases[i] + " ");
        }
        System.out.println("\n");
    }

    
    public void printNetworkParameteresGradients(){
        System.out.println("GRADfirstLayerWeights:");
        for(int i = 0; i < hiddenSize; i++){
            for(int j = 0; j < hiddenSize; j++) {
                System.out.print(GRADfirstLayerWeights[i][j] + " ");
            }
            System.out.println("");
        }
        System.out.println("\nGRADfirstLayerBiases:");
        for(int i = 0; i < hiddenSize; i++){
            System.out.print(GRADfirstLayerBiases[i] + " ");
        }
        System.out.println("");
        
        System.out.println("\nGRADsecondLayerWeights:");
        for(int i = 0; i < outputSize; i++){
            for(int j = 0; j < hiddenSize; j++) {
                System.out.print(GRADsecondLayerWeights[i][j] + " ");
            }
            System.out.println("");
        }
        System.out.println("\nGRADsecondLayerBiases:");
        for(int i = 0; i < outputSize; i++){
            System.out.print(GRADsecondLayerBiases[i] + " ");
        }
        System.out.println("\n");
    }
    
    public void printHiddenVector(){
        System.out.println("Hidden vector:");
        for(int i = 0; i < hiddenSize; i++){
            System.out.print(hiddenVector[i] + " ");
        }
        System.out.println("");
    }
    
    public void printOutputVector(){
        System.out.println("Output vector:");
        for(int i = 0; i < outputSize; i++){
            System.out.print(outputVector[i] + " ");
        }
        System.out.println("");
    }
    
    public void printTargetVector(double[] targetVector){
        System.out.println("Target vector:");
        for(int i = 0; i < outputSize; i++){
            System.out.print(targetVector[i] + " ");
        }
        System.out.println("");
    }
}
