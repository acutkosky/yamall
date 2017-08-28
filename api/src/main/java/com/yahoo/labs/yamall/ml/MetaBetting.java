// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.labs.yamall.ml;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.core.SparseVector;

import it.unimi.dsi.fastutil.ints.Int2DoubleMap;

@SuppressWarnings("serial")
public class MetaBetting implements Learner {
    private double initialWealth = 1;
    private transient double[] w = null;
    private transient double[] reward;
    private transient double[] sumStrongConvexities;
    private transient double[] sumMetaGradients;
    private transient double[] scaling;
    private transient double[] maxGrads;
    private Loss lossFnc;
    private int size_hash = 0;
    private int iter = 0;
    private boolean useWeightScaling = false;
    public MetaBetting(
            int bits) {
        size_hash = 1 << bits;
        sumStrongConvexities = new double[size_hash];
        reward = new double[size_hash];
        scaling = new double[size_hash];
        sumMetaGradients = new double[size_hash];
        maxGrads = new double[size_hash];
        w = new double[size_hash];
    }

    public void updateScalingVector(Instance sample ) {
        for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
            int key = entry.getIntKey();
            double value = Math.abs(entry.getDoubleValue());
            if (value > scaling[key] ) {
                scaling[key] = value;
            }

        }
    }

    private double getBeta(int key) {
        double unconstrainedOptimum = sumMetaGradients[key]/(sumStrongConvexities[key] + 0.25);
        if (unconstrainedOptimum > 0.5) {
            return 0.5;
        } else if (unconstrainedOptimum < -0.5) {
            return -0.5;
        } else {
            return unconstrainedOptimum;
        }
    }

    public void useWeightScaling(boolean flag) {
        useWeightScaling = flag;
    }

    public double update(Instance sample) {
        iter++;

        double pred = predict(sample);

        final double negativeGrad = lossFnc.negativeGradient(pred, sample.getLabel(), sample.getWeight());

        updateScalingVector(sample);
        if(Math.abs(negativeGrad) > 1e-8) {

            for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
                int key = entry.getIntKey();
                if(Math.abs(scaling[key]) > 1e-8) {
                    double x_i = entry.getDoubleValue();// /scaling[key] ;
                    double reward_i = reward[key];
                    double sumMetaGradients_i = sumMetaGradients[key];
                    double sumStrongConvexities_i = sumStrongConvexities[key];
                    double maxGrads_i = maxGrads[key];
                    double w_i = w[key];

                    maxGrads_i = Math.max(maxGrads_i, Math.abs(negativeGrad * x_i));

                    double negativeGrad_i = negativeGrad * x_i / maxGrads_i;
                    double beta_i = getBeta(key);

                    double metaGradient = negativeGrad_i / (1.0 + negativeGrad_i * beta_i);// / scaling[key];
                    double strongConvexity = negativeGrad_i * negativeGrad_i / ( ( 1.0 + Math.abs(negativeGrad_i)/2.0) * ( 1.0 + Math.abs(negativeGrad_i)/2.0)) ;// / scaling[key];


                    sumMetaGradients_i += metaGradient + beta_i * strongConvexity;
                    sumStrongConvexities_i += strongConvexity;
                    reward_i += getBeta(key) * (reward[key] + initialWealth) * negativeGrad_i;
                    maxGrads[key] = maxGrads_i;

                    reward[key] = reward_i;
                    sumMetaGradients[key] = sumMetaGradients_i;
                    sumStrongConvexities[key] = sumStrongConvexities_i;

                    //System.out.printf("negativeGrad: %f, metaGrad: %f, beta: %f, w: %f, reward: %f\n", negativeGrad_i, metaGradient, beta_i, w_i, reward_i);
                }
            }


            for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
                int key = entry.getIntKey();
                if(Math.abs(scaling[key]) > 1e-8)
                    w[key] = getBeta(key) * (reward[key] + initialWealth);// / scaling[key];
                if (useWeightScaling) {
                    w[key] /= scaling[key];
                }
                //System.out.printf("w: %f, reward: %f, beta: %f\n", w[key], reward[key], getBeta(key));
            }
            //System.out.printf("done an iteration!\n");
        }

        return pred;
    }


    public double predict(Instance sample) {
        return sample.getVector().dot(w);
    }


    public void setLoss(Loss lossFnc) {
        this.lossFnc = lossFnc;
    }

    public void setLearningRate(double eta) {
        this.initialWealth = eta;
    }

    public Loss getLoss() {
        return lossFnc;
    }

    public SparseVector getWeights() {
        return SparseVector.dense2Sparse(w);
    }

    public String toString() {
        String tmp = "Using MetaBetting optimizer\n";
        tmp = tmp + "Initial learning rate = " + initialWealth + "\n";
        tmp = tmp + "Loss function = " + getLoss().toString();
        return tmp;
    }

    private void writeObject(ObjectOutputStream o) throws IOException {
        o.defaultWriteObject();
        o.writeObject(SparseVector.dense2Sparse(w));
        o.writeObject(SparseVector.dense2Sparse(reward));
        o.writeObject(SparseVector.dense2Sparse(sumMetaGradients));
        o.writeObject(SparseVector.dense2Sparse(sumStrongConvexities));
        o.writeObject(SparseVector.dense2Sparse(scaling));
    }

    private void readObject(ObjectInputStream o) throws IOException, ClassNotFoundException {
        o.defaultReadObject();
        w = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        reward = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        sumMetaGradients = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        sumStrongConvexities = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        scaling = ((SparseVector) o.readObject()).toDenseVector(size_hash);
    }


}