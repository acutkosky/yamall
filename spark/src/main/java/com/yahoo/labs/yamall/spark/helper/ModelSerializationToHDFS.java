package com.yahoo.labs.yamall.spark.helper;

import com.yahoo.labs.yamall.ml.IOLearner;
import com.yahoo.labs.yamall.ml.Learner;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import java.io.File;
import java.io.IOException;

/**
 * Created by busafekete on 8/29/17.
 */
public class ModelSerializationToHDFS {
    public static void saveModel(String dir, String fname, Learner learner) throws IOException {
        FileDeleter.delete(new File(dir + fname));
        IOLearner.saveLearner(learner, fname);

        // copy output to HDFS
        FileSystem fileSystem = FileSystem.get(new Configuration());
        fileSystem.moveFromLocalFile(new Path(fname), new Path(dir));

    }


    private static final String MODEL_BIN = "model.bin";
    public static Learner loadModel(String dir, String fname) throws IOException {
        // move model to the node
        FileSystem fileSystem = FileSystem.get(new Configuration());
        fileSystem.copyToLocalFile(new Path(dir + fname), new Path(MODEL_BIN));

        Learner learner = IOLearner.loadLearner(MODEL_BIN);
        return learner;
    }


}
