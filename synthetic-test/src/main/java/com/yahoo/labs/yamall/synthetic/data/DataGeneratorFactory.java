package com.yahoo.labs.yamall.synthetic.data;

import java.util.Properties;

/**
 * Created by busafekete on 8/15/17.
 */
public class DataGeneratorFactory {
    public static class DataPair {
        public DataGenerator train = null, test=null;
        public DataPair(DataGenerator train, DataGenerator test) {
            this.train = train;
            this.test = test;
        }
    }

    public static DataPair getDataGenerator(Properties properties ) {
        int trainSize = Integer.parseInt(properties.getProperty("tain_size", "5000000"));
        int testSize = Integer.parseInt(properties.getProperty("test_size", "50000"));
        int dim = 1000;
        int sparsity = 100;
        int seed = Integer.parseInt(properties.getProperty("seed", "0"));

        String dataType = properties.getProperty("data_type", "default");
        DataGenerator train = null;
        DataGenerator test = null;
        if (dataType.compareToIgnoreCase("default") == 0) {
            train = new DataGeneratorNormal(trainSize, dim, sparsity);
            test = train.copy();
            ((DataGeneratorNormal) test).setNum(testSize);
        } else if (dataType.compareToIgnoreCase("file") == 0) {
            String trainFile = properties.getProperty("train_file");
            train = new DataGeneratorFromFile(trainFile);

            String testFile = properties.getProperty("test_file");
            test = new DataGeneratorFromFile(testFile);
        } else if (dataType.compareToIgnoreCase("gzfile") == 0) {
            String trainFile = properties.getProperty("train_file");
            train = new DataGeneratorFromZippedFile(trainFile);

            String testFile = properties.getProperty("test_file");
            test = new DataGeneratorFromZippedFile(testFile);
        }


        System.out.println( "Train size: " + trainSize );
        System.out.println( "Test file: " + testSize );

        System.out.println( "Dim: " + dim );
        System.out.println( "Sparsity: " + sparsity );

        return new DataPair(train, test);
    }
}
