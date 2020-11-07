import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
/**
 * This class contains a static method to parse csv files and 
 * generates an matrix containing all the training data in the file. 
 * Here, a matrix is implemented as a list of double arrays. 
 * @author Hyejin Jenny Yeon
 */
public class DataParser {
	
    /**
     * This method parses csv file and returns a matrix as ArrayList<double[]>.
     * @param filePath is the path of the file. 
     * @return ArrayList<double[]> a matrix containing all the training data
     * @throws FileNotFoundException
     * @throws IOException
     */
    public static ArrayList<double[]> parseTrainingRecords(String filePath) throws FileNotFoundException, IOException {
    	ArrayList<double[]> data = new ArrayList<double[]>();
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line = "";
        reader.readLine();
        while ((line = reader.readLine()) != null) {
            String[] stringValues = line.split(",");
            double[] doubleValues = new double[stringValues.length];
           	for (int i = 0 ; i < stringValues.length ; i ++) {
               	doubleValues[i] = Double.parseDouble(stringValues[i]);           	
               	}
            data.add(doubleValues);
        }
        reader.close();
    return data;
    }
}
