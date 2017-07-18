package com.yahoo.labs.yamall.ml;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.core.SparseVector;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

/**
 * Created by busafekete on 7/9/17.
 */
public class SVRG implements Learner {
    private boolean averaging = false;

    private static final int GATHER_GRADIENT = 1;
    private static final int UPDATE_GRADIENT = 2;

    private double eta = .01;
    private int step = 500;
    private double lambda = .01;

    private int backCounter = 0;
    private int gradStep = 0;

    private transient double[] w;
    private transient int[] last_updated; //records when each index was last updated for proper regularization

    private transient double[] Gbatch;
    private transient double[] w_prev;
    private transient double[] w_avg;

    private double N = 0;
    private Loss lossFnc;
    private double iter = 0;
    private int size_hash = 0;

    private int gatherGradIter = 0;

    private int state = 2;

    public SVRG(
            int bits) {
        size_hash = 1 << bits;
        w = new double[size_hash];

        lambda = 0.1/Math.sqrt(step);

        Gbatch = new double[size_hash];
        w_prev = new double[size_hash];
        // we can get rif of this by using online update
        w_avg = new double[size_hash];
    }

    public void setLoss(Loss lossFnc) {
        this.lossFnc = lossFnc;
    }

    public void setLearningRate(double eta) {
        this.eta = eta;
    }

    public void setRegularizationParameter(double lambda) {
        this.lambda = lambda;
    }

    public void doAveraging() { this.averaging = true; }

    public void setStep(int step) {
        this.step = step;
    }

    private double accumulateGradient( Instance sample ) {
        gatherGradIter++;

        double pred = predict(sample);

        final double grad = -lossFnc.negativeGradient(pred, sample.getLabel(), sample.getWeight());

        if (Math.abs(grad) > 1e-8) {
            sample.getVector().addScaledSparseVectorToDenseVector(Gbatch, grad);
        }
        return pred;
    }

    public double gradStep( Instance sample ){
        double pred = 0;
        double pred_prev = 0;
        gradStep++;

        if (lambda != 0.0) {
            //this loop lazily applies several steps of SGD on the regularizer in a row.
            for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
                int key = entry.getIntKey();
                double decay_rate = 1.0 - lambda * eta;
                int missed_steps = gradStep - last_updated[key] - 1
                scaling = Math.pow(decay_rate, missed_steps);
                w[key] = w_prev[key] + scaling * (w[key] - w_prev[key]);

                //update average properly (update to gradStep - 1th average).
                if (averaging) {
                    double sum_decay_powers = (decay_rate - math.pow(decay_rate, missed_steps+1)) / (1.0 - decay_rate);
                    w_avg[key] = (last_update[key] * w_avg[key] +  sum_decay_powers * w[key])/(gradStep - 1);
                }
                last_updated[key] = gradStep;
            }
        }

        pred = predict(sample);
        pred_prev = predict_prev(sample);

        final double grad = lossFnc.negativeGradient(pred, sample.getLabel(), sample.getWeight());
        final double grad_prev = lossFnc.negativeGradient(pred_prev, sample.getLabel(), sample.getWeight());
        final double grad_diff = grad_prev - grad;

        
        if (Math.abs(grad) > 1e-8) {
            sample.getVector().addScaledSparseVectorToDenseVector(w, eta * grad_diff);

            if (averaging) {
                for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
                    int key = entry.getIntKey();
                    w_avg[key] += (w[key] - w_avg[key])/((double)gradStep);
                }
            }
        }
        return pred;
    }

    private void initGatherState() {

        if (lambda != 0.0) {
            for (int i=0; i < size_hash; i++) {
                if (last_update[i] < gradStep) {
                    double decay_rate = 1.0 - lambda * eta;
                    int missed_steps = gradStep - last_updated[key]
                    scaling = Math.pow(decay_rate, missed_steps);
                    w[i] = w_prev[i] + scaling * (w[i] - w_prev[i]);
                    //update average properly (update to gradStep th average).
                    if (averaging) {
                        double sum_decay_powers = (decay_rate - math.pow(decay_rate, missed_steps+1)) / (1.0 - decay_rate);
                        w_avg[key] = (last_update[key] * w_avg[key] +  sum_decay_powers * w[key])/(gradStep);
                    }
                }
                last_updated[i] = 0;
            }
        }

        if (this.averaging ) {
            for (int i = 0; i < size_hash; i++) w[i] = w_avg[i];
        }

        for (int i=0; i < size_hash; i++ ) w_prev[i] = w[i];
        for (int i=0; i < size_hash; i++ ) Gbatch[i] = 0;
        gatherGradIter = 0;
    }

    private void normalizeBatchGradient() {
        for (int i=0; i < size_hash; i++ ) Gbatch[i] /= (double)gatherGradIter;

        if (this.averaging ) {
            for (int i = 0; i < size_hash; i++) w_avg[i] = w[i];
        }

        gradStep = 0;
    }

    public double update(Instance sample) {
        iter++;

        alterState();

        double pred = 0;
        if (state == SVRG.GATHER_GRADIENT) {
            pred = this.accumulateGradient(sample);
        } else if (state == SVRG.UPDATE_GRADIENT) {
            pred = this.gradStep(sample);
        }


        return pred;
    }

    private void alterState() {
        backCounter--;
        if ( backCounter <= 0  ) {
            if (state == SVRG.GATHER_GRADIENT ){ // switch to update parameters
                backCounter = step;
                normalizeBatchGradient();
                state = SVRG.UPDATE_GRADIENT;
            } else if ( state == SVRG.UPDATE_GRADIENT ) { // switch to gather gradient
                backCounter = step;
                initGatherState();
                state = SVRG.GATHER_GRADIENT;
            }
        }
    }

    public double predict(Instance sample) {
        return sample.getVector().dot(w);
    }

    public double predict_prev(Instance sample) {
        return sample.getVector().dot(w_prev);
    }

    public Loss getLoss() {
        return lossFnc;
    }

    public SparseVector getWeights() {
        return SparseVector.dense2Sparse(w);
    }

    public String toString() {
        String tmp = "Using SVRG\n";
        tmp = tmp + "Initial learning rate = " + eta + "\n";
        tmp = tmp + "Loss function = " + getLoss().toString();
        return tmp;
    }

    private void writeObject(ObjectOutputStream o) throws IOException {
        o.defaultWriteObject();
        o.writeObject(SparseVector.dense2Sparse(w));
        o.writeObject(SparseVector.dense2Sparse(Gbatch));
        o.writeObject(SparseVector.dense2Sparse(w_prev));
    }

    private void readObject(ObjectInputStream o) throws IOException, ClassNotFoundException {
        o.defaultReadObject();
        w = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        Gbatch = ((SparseVector) o.readObject()).toDenseVector(size_hash);
        w_prev = ((SparseVector) o.readObject()).toDenseVector(size_hash);
    }
}
