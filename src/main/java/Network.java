/**
 * @author RusFortunat
 */

public class Network {
    private int inputSize;
    private int hiddenSize;
    private int outputSize;
    private double[][] firstLayerWeights;
    private double[] firstLayerBiases;
    private double[][] secondLayerWeights;
    private double[] secondLayerBiases;
    
    // at the moment just 1 hidden layer, generalization will come later
    public Network(int inputSize, int hiddenSize, int outputSize){
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.firstLayerWeights = new double[inputSize][hiddenSize];
        this.firstLayerBiases = new double[hiddenSize];
        this.secondLayerWeights = new double[hiddenSize][outputSize];
        this.secondLayerBiases = new double[outputSize];
    }
    
    public double[] forward(double[] input){
        double[] predicted = new double[outputSize];
        
        return predicted;
    }
    
}
