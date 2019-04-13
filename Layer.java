/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnet2;

import static java.lang.Math.sqrt;
import java.util.Random;

/**
 *
 * @author James
 */
public class Layer {
    int type;
    int func;
    Node[] nodes;
    double[][] weights;
    double[] outputs;
    double[][] prev;
    //The Layer is constructed with an integer for the type of layer it is, the number of nodes it must contain
    //and integer representing the activation function of those nodes, and the number of inputs it has
    public Layer(int type, int numNodes, int func, int numInputs) {
        nodes = new Node[numNodes];
        this.type = type;
        this.func = func;
        //here it creates all the nodes it needs
        for(int i = 0; i < numNodes; i++) {
            nodes[i] = new Node(func);
        }
        Random ran = new Random();
        weights = new double[numNodes][numInputs];
        //here is randomly generates weights
        for(int i = 0; i < numNodes; i++) {
            for(int j = 0; j < numInputs; j++) {
                weights[i][j] = (ran.nextDouble() - 0.5);
            }
        }
        prev = new double[numNodes][numInputs];
        //here is randomly generates weights
        for(int i = 0; i < numNodes; i++) {
            for(int j = 0; j < numInputs; j++) {
                prev[i][j] = 0;
            }
        }
        outputs = new double[numNodes];
    }
    //This function runs the activation functions on all the nodes in the layer
    //by using and array of doubles to get a weighted sum of the inputs to pass to the nodes for their activation function
    //if the nodes are using gaussian function it instead calculates the magnitude
    public void activate(double[] output) {
        double sum = 0;
        if(type != 3) {
            for(int i = 0; i < nodes.length; i++) {
                for(int j = 0; j < output.length; j++) {
                    sum += output[j] * weights[i][j];
                }
                outputs[i] = nodes[i].activate(sum+1);
                sum = 0;
            }
        } else {
            for(int i = 0; i < nodes.length; i++) {
                for(int j = 0; j < output.length; j++) {
                    sum += output[j] * weights[i][j];
                }
                outputs[i] = nodes[i].activate(sum+1);
                if(outputs[i] >= 0.5) {
                    outputs[i] = 1;
                } else {
                    outputs[i] = 0;
                }
                sum = 0;
            }
        }
    }
    //This function sets sigma values in all nodes in the layer, it uses a value inputed by the user to determine how to set the sigmas
    public void setSigmas(int a) {
        if(a == 1) {
            for(int i = 0; i < nodes.length; i++) {
                nodes[i].setSigma(25);
            }
        } else if(a == 2) {
            for(int i = 0; i < nodes.length; i++) {
                if((i % 2) == 0) {
                    nodes[i].setSigma(25);
                } else {
                    nodes[i].setSigma(15);
                }
            }
        } else if(a == 3) {
            Random ran = new Random();
            for(int i = 0; i < nodes.length; i++) {
                int b = 1 + ran.nextInt(100);
                nodes[i].setSigma(b);
            }
        }
    }
    //This function is for running backpropagation on the output layer specifically
    public void backpropO(double d, double[] inputs) {
        double temp;
        for(int i = 0; i < nodes.length; i++) {
            temp = (d - nodes[i].output);
            nodes[i].setDelta(temp);
        }
        
        for(int j = 0; j < nodes.length; j++) {
            for(int i = 0; i < inputs.length; i++) {
                double a = nodes[j].delta;
                double b = inputs[i];
                weights[j][i] += 0.008 * a * b + 0.5 * prev[j][i];
                prev[j][i] = 0.008 * a * b + 0.5 * prev[j][i];
            }
        }
    }
    //This function is for running backpropagation on the hidden layers
    public void backpropH(Layer output, double[] inputs) {
        double temp;
        for(int j = 0; j < nodes.length; j++) {
            temp = 0.0;
            for(int i = 0; i < output.nodes.length; i++) {
                temp += output.weights[i][j] * output.nodes[i].delta;
            }
            nodes[j].setDelta(-1*nodes[j].output * (1 - nodes[j].output) * temp);
        }
        
        for(int j = 0; j < nodes.length; j++) {
            for(int i = 0; i < inputs.length; i++) {
                double a = nodes[j].delta;
                weights[j][i] += 0.008 * a + 0.5 * prev[j][i];
                prev[j][i] = 0.008 * a + 0.5 * prev[j][i];
            }
        }
    }
    
}
