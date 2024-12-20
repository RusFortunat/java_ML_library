/**
 * I don't know the structure of the data files, so I followed this example of reading MNIST file:
 * https://github.com/turkdogan/mnist-data-reader
 */
import java.io.*;

// at the moment just 1 hidden layer, generalization will come later
public class MNISTImageData {
    private String imagesPath;
    private String labelsPath;
    private int datasetSize;
    private double[][] images; // first index for image number and second represents image itself casted onto vector
    private int[] labels; 

    public MNISTImageData(String imagesPath, String labelsPath, int trainDatasetSize, int imageSize){
        this.imagesPath = imagesPath;
        this.labelsPath = labelsPath;
        this.datasetSize = trainDatasetSize;
        this.images = new double[trainDatasetSize][imageSize];
        this.labels = new int[trainDatasetSize];
    }

    public void loadImageData() throws IOException{
        DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(imagesPath)));
        int magicNumber = dataInputStream.readInt();
        int numberOfItems = dataInputStream.readInt();
        int nRows = dataInputStream.readInt();
        int nCols = dataInputStream.readInt();
        /*
	System.out.println("magic number is " + magicNumber);
        System.out.println("number of items is " + numberOfItems);
        System.out.println("number of rows is: " + nRows);
        System.out.println("number of cols is: " + nCols);
        */
        DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(labelsPath)));
        int labelMagicNumber = labelInputStream.readInt();
        int numberOfLabels = labelInputStream.readInt();
        /*
	System.out.println("labels magic number is: " + labelMagicNumber);
        System.out.println("number of labels is: " + numberOfLabels);
        */

        assert datasetSize == numberOfLabels;

        for(int i = 0; i < datasetSize; i++) {
            labels[i] = labelInputStream.readUnsignedByte();
            double[] image = new double[nRows*nCols];
			
            for (int r = 0; r < nRows; r++) {
                for (int c = 0; c < nCols; c++) {
                    image[r*nCols + c] = dataInputStream.readUnsignedByte() / 255.0;
                }
            }
            images[i] = image;
            //printImage(i, nRows, nCols); // check if the data is loaded properly -- words so far i can tell
        }
        dataInputStream.close();
        labelInputStream.close();
    }

    public double[][] getImages(){
        return images;
    }

    public int[] getLabels(){
        return labels;
    }
	
    public void printImage(int image, int nRows, int nCols){
        System.out.println("Image: " + image);
        for(int row = 0; row < nRows; row++){
            for(int col = 0; col < nCols; col++){
                double pixel = images[image][row*nCols + col];
                System.out.print(pixel + " ");
            }
            System.out.print("\n");
        }
        System.out.println("");
    }


}
