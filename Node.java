/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnet2;

import static java.lang.Math.exp;

/**
 *
 * @author James
 */
public class Node {
    double output;
    int func; //variable for specifying activation function
    double sum;
    int sigma;
    double delta;
    //Constructor
    public Node(int func) {
        this.func = func;
    }
    //Function that selects activation function based on value passed during initialization
    public double activate(double weightedSum) {
        switch(func) {
            case 1:
                linear(weightedSum);
                break;
            case 2:
                sigmoidal(weightedSum);
                break;
            case 3: 
                gaussian(weightedSum);
                break;
        }
        return output;
    }
    //Linear function for output layer
    private void linear(double weightedSum) {
        output = 1 * weightedSum;
    }
    //Sigmoidal function, generate value between 0 and 1
    private void sigmoidal(double weightedSum) {
        output = 1 / (1 + exp(weightedSum));
    }
    //Gaussian function used in RBF using sigma set shortly after initialization
    private void gaussian(double magnitude) {
        output = exp(-1 * ((magnitude * magnitude)/(2 * (sigma * sigma))));
    }
    //Function to pass sigma value for use in gaussian function
    public void setSigma(int sigma) {
        this.sigma = sigma;
    }
    //Function to pass delta value used during backpropigation
    public void setDelta(double delta) {
        this.delta = delta;
    }
}
