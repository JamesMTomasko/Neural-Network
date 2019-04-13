/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnet;

import static java.lang.Math.exp;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

/**
 *
 * @author James
 */
public class Node {
    private int func;
    private double[] in;
    private double out = 0;
    private double delta;
    double sigma = 0;
    private Layer layer;
    
    public Node(int func, Layer layer) {
        this.func = func;
        this.layer = layer;
    }
    
    public double activate(double[] in, double[] weights) {
        this. in = in;
        if(func == 1) {
            out = sigmoidal(weights);
        } else if(func == 2) {
            out = gaussian();
        } else if(func == 3) {
            out = linear(weights);
        }
        return out;
    }
    
    public double getOutput() {
        return out;
    }
    
    public double linear(double[] weights) {
        double temp = 0;
        for(int i = 0; i < in.length; i++) {
            temp += in[i] * weights[i];
        }
        temp += 1;
        return temp * 2;
    }
    
    public double sigmoidal(double[] weights) {
        double temp = 0;
        for(int i = 0; i < in.length; i++) {
            temp += in[i] * weights[i];
        }
        temp += 1;
        double val = 1 / (1 + exp(temp));
        return val;
    }
    
    public double gaussian() {
        double z;
        double dif = 0;
        for(int i = 0; i < in.length; i++) {
            dif += in[i] * in[i];
        }
        z = exp(-1 * (dif/(2 * (sigma * sigma))));
        return z;
    }
    
    public void setDelta(double d) {
        delta = d;
    }
    
    public double getDelta() {
        return delta;
    }
    
    public void setSigma(double sigma) {
        this.sigma = sigma;
    }
    
    public void setOut(double out) {
        this.out = out;
    }
}
