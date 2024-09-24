/**
 * @author RusFortunat
 */
import java.util.concurrent.ThreadLocalRandom;

// at the moment just 1 hidden layer, generalization will come later
public class Network {
    private int inputSize;
    private int hiddenSize;
    private int outputSize;
    private double[][] firstLayerWeights;
    private double[] firstLayerBiases;
    private double[][] secondLayerWeights;
    private double[] secondLayerBiases;
    
    // initialize a fully-connected neural network with random weights and biases
    public Network(int inputSize, int hiddenSize, int outputSize){
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.firstLayerWeights = new double[inputSize][hiddenSize];
        this.firstLayerBiases = new double[hiddenSize];
        this.secondLayerWeights = new double[hiddenSize][outputSize];
        this.secondLayerBiases = new double[outputSize];
        
        double rangeW1 = 1.0/inputSize; // it is a good practice to distribte 
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
    
    // implements feed-forward propagation with reLU activation
    public double[] forward(double[] input){
        double[] predicted = new double[outputSize];
        
        return predicted;
    }
    
}
