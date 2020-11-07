import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * This class can perform a logistic regression on a training data set.
 * There are public methods to help one to perform logistic regression.  
 * The main method shows a suggested use for this program. 
 * @author Hyejin Jenny Yeon
 */
public class LogisticRegression {
	
	/**
	 * The main method makes an instance of the class LogisticRegression and runs 
	 * various methods in the class. This may run on a different class. 
	 * The cost function here is the log-likelihood function. 
	 * @param args
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public static void main(String[] args) throws FileNotFoundException, IOException {
		// File Requirement: each item should be in (label, values of features seperated by ",") 
		LogisticRegression test = new LogisticRegression("titanic_data.csv");
		//test.crossValidation(200);
		//int[] subfeatures = {1,2,3,4,5,6}; // All 6 features are used
		//test.subFeatures(subfeatures);
		//test.gradientDescent(1000000, test.stepSize);
		test.fixThetas();
		ArrayList<double[]> infoMatrix = test.hessianMatrix();
		System.out.println("Info Matrix: ");
		for (double[] d : infoMatrix) {
			System.out.print("{");
			for (int i = 0 ; i < d.length ; i++) {
				if (i < d.length-1) System.out.printf("%.4f, ",d[i]);
				else System.out.printf("%.9f",d[i]);
			}
			System.out.print("}, ");
			System.out.println();
			
		}
		//Print out the coefficients:
		System.out.println("Constant Term, theta zero: "+ test.b +System.lineSeparator()+"Thetas: ");
		for (double d : test.thetas) System.out.println(d);
		System.out.println("============================");
		ArrayList<double[]> myData = new ArrayList<double[]>();
		
		//the first position is y hat, if unknown, just put any number
		myData.add(new double[]{1,3,1,28,0,0,8.05}); 
		myData.add(new double[]{1,3,0,28,0,0,8.05});
		test.changeTestSet(myData);
		test.gradientDescentTest(true);
		System.out.println("Cost is: " + test.costFunction());

	}
	
	// Fields associated with the class LogisticRegression 
    private Random random = new Random();
	private double[] thetas; //Weights
	private double b; //bias or zeroth entry of theta
    private double stepSize; 
    // Both trainingData and testing follows the form
    // (label: either zero or 1, features)
    // testing can also be used for a new feature vector
    // by simply letting the first entry to be any number. 
    private List<double[]> trainingData;
    private List<double[]> testing; 
    private double prevCost;
    private double newCost;
    private Double[] as; // ai's, which is the value of Sigmoid functions 
                         // applied to all training data.
    private double epsilon; //convergence tolerance
    
    /**
     * Constructor initializes the training data matrix using a given
     * file and other fields to their default values
     * @param trainingDataPath
     * @throws FileNotFoundException
     * @throws IOException
     */
    public LogisticRegression (String trainingDataPath) throws FileNotFoundException, IOException {
		// Index 0 = survive or not, Index 1-6 = features
    	trainingData = DataParser.parseTrainingRecords(trainingDataPath);
    	// Thetas and b are initialized with random values
    	thetas = new double[trainingData.get(0).length-1];
    	b = random.nextDouble();
    	for (int i = 0; i < thetas.length; i++) thetas[i] = (2.0*random.nextDouble()-1.0); 
    	prevCost = 0.0;
        newCost = 0.0;
        // default values for epsilon and step size
        epsilon = 0.0000000001;
        stepSize = 0.01;     
    }
    
    /**
     * This method can be called if one wants to use only selected features 
     * for logistic regression. 
     * @param featureNumbers list of features to be used
     */
    public void subFeatures(int[] featureNumbers){
    	ArrayList<double[]> subFeatures = new ArrayList<double[]>();
    	for (double[] d : this.trainingData) {
    		double[] subset = new double[featureNumbers.length+1];
    		subset[0]=d[0]; //labels copies
    		for (int i = 1 ; i < subset.length ; i++) {
    			subset[i]=d[featureNumbers[i-1]];
    		}
    				
    		subFeatures.add(subset);
    	}
    	this.modifyTrainingData(subFeatures);
    }
    
    /**
     * This cuts the training set into two sets, one being the testing set
     * and the rest of being the training set. This method is not fully 
     * implemented. 
     * @param cutoff is the index of the item in the data matrix that 
     *        the data set should split into
     */
    public void crossValidation(int cutoff) {
    	this.testing = trainingData.subList(0, cutoff);
    	trainingData = trainingData.subList(cutoff, trainingData.size());
    }
    
    /*
     * This method computes Sigmoid function value. 
     * Sigmoid(x) = 1 / ( 1 + exp(-x) )
     * (I end up not using this b/c there was no need to clutter stack by
     * calling this methods)
     */
    public double sigmoid(double x){
    	return 1.0 / (1.0 + Math.exp(-1.0 * (x)));
    }
    
    /*
     * Computes derivative of Sigmoid function. 
     * Derivative of Sigmoid is also (1-A)(A) if we let A = 1.0 / (1.0 + exp(-x)))
     * I end up not using this for the same reason as the method sigmoid(double x)
     */
    public double derivativeSigmoid(double x){
    	return Math.exp(-1.0*(x)) / Math.pow(1.0 + Math.exp(-1.0 * (x)),2) ;
    }
    
    /**
     * This method implements gradient descent(or accent).  
     * @param maxNumIterations is the max number of iterations allowed
     * @param initialStepSize is the initial step size for gradient descent
     */
    public void gradientDescent(int maxNumIterations, double initialStepSize) {
        // Gradient descent step
        for (int iteration = 1; ; iteration++) {
        	// Learning rate (=step size) decreases for each iterations
        	double stepSize = 0.2*initialStepSize/Math.sqrt(iteration);
        	
            // Calculate a_i array
            Double[] a = new Double[this.trainingData.size()];
            for (int i = 0; i < this.trainingData.size(); i++) {
                double sum_wx = 0;
                for (int j = 0; j < thetas.length; j++) {
                    sum_wx += thetas[j] * this.trainingData.get(i)[j+1]; //1st index is the label 0 or 1
                }
                a[i] = 1.0 / (1.0 + Math.exp(-1.0 * (sum_wx + b)));
            }

            // Update weights 
            for (int j = 0; j < thetas.length; j++) {
                double w_temp = 0;
                for (int i = 0; i < this.trainingData.size(); i++) {
                    w_temp += (a[i] - this.trainingData.get(i)[0]) * this.trainingData.get(i)[j+1];
                }
                thetas[j] = thetas[j] - stepSize * w_temp;            }
            
            // Update bias
            double b_temp = 0;
            for (int i = 0; i < trainingData.size(); i++) {
                b_temp += (a[i] - trainingData.get(i)[0]);
            }
            b -= stepSize * b_temp;

            // Calculate cost function to check convergence
            prevCost = newCost;
            newCost = 0.0;
            for (int i = 0; i < trainingData.size(); i++) {
                if (trainingData.get(i)[0] == 0.0) {
                    if (a[i] > 0.9999999) {
                    	newCost += 1000.0; // something large so that
                    	                   // the iteration does not stop
                    }
                    else {
                    	// Note that 1/(1+e^A)=1-1/(1+e^(-A))
                    	newCost -= Math.log(1 - a[i]);
                    }
                }
                else if (trainingData.get(i)[0] == 1.0) {
                    if (a[i] < 0.0000001) newCost += 1000.0;
                    else newCost -= Math.log(a[i]);
                }
            }

            // Check for convergence
            double convergence = Math.abs(newCost - this.prevCost);
            if (convergence < this.epsilon) {
            	System.out.println("Process completed in "+ iteration +" iterations");
            	break;
            }
            else if (iteration > maxNumIterations) { // termination condition
                System.out.println("Reached the maximum number of iterations. "
                        + "Consider changing the stepSize");
                break;
            }
        }
    }
    
    /**
     * The helper method to change training data
     * @param newData
     */
    private void modifyTrainingData(List<double[]> newData) {
    	this.trainingData = newData;    	
    	this.thetas=new double[newData.get(0).length-1];
    }
    
    /**
     * This method tests if the logistic regression correctly 
     * guess the label or not. This method can also simply be used to 
     * print the value of sigmod(linear combination of thetas with 
     * the data). If printing is turned on, it also prints overall
     * accuracy result. 
     * @param printResult prints result if true, and does not if false
     */
    public void gradientDescentTest(boolean printResult) {
    	this.as = new Double[this.testing.size()];
        for (int i = 0; i < this.testing.size(); i++) {
            double sum_wx = 0.0;
            for (int j = 0; j < thetas.length; j++) {
                sum_wx += this.thetas[j] * this.testing.get(i)[j+1]; 
            }
            double temp = 1.0 / (1.0 + Math.exp(-1.0 * (sum_wx + b)));
            if (printResult) System.out.println(temp);
            if (temp > 0.5) {
                as[i] = 1.0;
            }
            else {
            	as[i] = 0.0;
            }

        }
        if (printResult) {
    		int counter=0;
    		System.out.print("Expected Labels: ");
    		for (int i = 0 ; i < this.testing.size() ; i ++) {
    			System.out.print(this.testing.get(i)[0]+", ");
    			if(this.testing.get(i)[0]==this.as[i]) {				
    				counter++;
    			}
    		}
    		System.out.println(System.lineSeparator()+ "Result: "+ counter 
    				+ " correct out of " + this.testing.size());
    		
        }
    }
    /**
     * The method allows the user to change step size. 
     * @param newSize
     */
    public void changeStepSize(double newSize) {
    	this.stepSize = newSize;
    }
    /**
     * The method allows the user to change the test set. 
     * @param matrix containing test set 
     */
    public void changeTestSet(List<double[]> d) {
    	this.testing = d;
    }
    /**
     * The method allows the user to change the value of epsilon
     * hence allowing the user to change the convergence criteria.
     * @param step size
     */
    public void changeEpsilon(double d) {
    	this.epsilon = d;
    }
    /**
     * This computes the cost function with the current value of thetas. 
     * @return the value of cost function
     */
    public double costFunction() {
    	double cost = 0.0;
    	double linearCombo = this.b;
    	for (double[] d : this.trainingData) {
    		for (int i = 1 ; i < this.thetas.length+1 ; i ++) {
        		linearCombo += d[i]*this.thetas[i-1];
    		}
    		if (d[0] == 0.0) {
    			cost +=Math.log(1/(1+Math.exp(linearCombo)));
    		}
    		else {
    			cost +=Math.log(1/(1+Math.exp(-1*linearCombo)));
    		}
    		linearCombo = this.b;
    	}
    	
    	return cost;
    }
    /**
     * This method allows the user to fix the value of thetas to be
     * the value found using this program's gradient descent 
     * with epsilon = 0.0000000001. 
     * This may be useful if one wants to use the program 
     * only to test their data. 
     */
    public void fixThetas() {
    	this.b=2.5384257469974516;
    	this.thetas = new double[] {
    			-1.1772586350304417
    			,2.757213061941336
    			,-0.04345604758946768
    			,-0.4017737515368076
    			,-0.10652114893035007
    			,0.002788941470705457};
    }
    
    /**
     * This computes the Hessian matrix for the cost function.
     * @return
     */
    public ArrayList<double[]> hessianMatrix() {
    	ArrayList<double[]> matrix= new ArrayList<double[]>();
    	int dim = this.thetas.length+1;
    	int n = this.trainingData.size();
		double ijEntry = 0.0;
		double[] rowi = new double[dim];
		for (int i = 0 ; i < dim ; i++) {
			for (int j = 0 ; j < dim ; j++ ) {
				for (int k = 0 ; k < n ; k++) {
					double[] datak = this.trainingData.get(k);
					double temp1 = this.linearCombo(k);
					double temp2 = 0.0;
					if (i==0&&j==0) {
						temp2 = Math.exp(-1*temp1);
					}
					else if (i==0) {
						temp2 = datak[j]*Math.exp(-1*temp1);
					}
					else if (j==0) {
						temp2 = datak[i]*Math.exp(-1*temp1);
					}
					else {
						temp2 = datak[i]*datak[j]*Math.exp(-1*temp1);
					}										 
					ijEntry += temp2/Math.pow(1+Math.exp(-1*temp1), 2);
				}
				rowi[j] = ijEntry;
				ijEntry = 0.0;
			}
			matrix.add(rowi);
			rowi = new double[dim];
		}
    	return matrix;
    }
    
    /**
     * A helper method to compute linear combination of the data
     * with the theta + bias values. 
     * @param i
     * @return
     */
    private double linearCombo(int i) {
    	double linearCombo = this.b;
    	double[] datai = this.trainingData.get(i);
		for (int j = 1 ; j < this.thetas.length+1 ; j ++) {
    		linearCombo += datai[j]*this.thetas[j-1];
		}
    	return linearCombo;
    }
}
