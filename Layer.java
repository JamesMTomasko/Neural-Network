/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnet;

import java.util.Random;

/**
 *
 * @author James
 */
public class Layer {
    double[][] weights;
    double[] outputs;
    Node[] nodes;
    Node[] inputs;
    Layer prev;
    
    public Layer(Node[] inputs, int numNodes, int func, Layer prev) {
        weights = new double[numNodes][inputs.length];
        outputs = new double[numNodes];
        this.nodes = new Node[numNodes];
        this.inputs = inputs;
        this.prev = prev;
        for(int i = 0; i < numNodes; i++) {
            nodes[i] = new Node(func, this);
        }
        randomWeights();
    }
    
    public Layer( int numNodes) {
        outputs = new double[numNodes];
        this.nodes = new Node[numNodes];
        for(int i = 0; i < numNodes; i++) {
            nodes[i] = new Node(1, this);
        }
    }
    
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
                double b = (ran.nextDouble() * 100) +1;
                nodes[i].setSigma(b);
            }
        }
    }
    
    public void activate() {
        double[] in = prev.getOutputs();
        for (int j = 0; j < nodes.length; j++) {
            outputs[j] = nodes[j].activate(in, weights[j]);
        }
    }
    
    public void backpropO(double[] outs) {
        double temp;
        for(int i = 0; i < nodes.length; i++) {
            temp = (outs[i] - nodes[i].getOutput());
            nodes[i].setDelta(temp);
        }
        
        for(int j = 0; j < nodes.length; j++) {
            for(int i = 0; i < inputs.length; i++) {
                double a = nodes[j].getDelta();
                double b = inputs[i].getOutput();
                weights[j][i] += 0.9 * a * b;
            }
        }
    }
    
    public void backpropH(Layer output) {
        double temp;
        for(int j = 0; j < nodes.length; j++) {
            temp = 0.0;
            for(int i = 0; i < output.nodes.length; i++) {
                temp += output.weights[i][j] * output.nodes[i].getDelta();
            }
            nodes[j].setDelta(-1*nodes[j].getOutput() * (1 - nodes[j].getOutput()) * temp);
        }
        
        for(int j = 0; j < nodes.length; j++) {
            for(int i = 0; i < inputs.length; i++) {
                double a = nodes[j].getDelta();
                double b = inputs[i].getOutput();
                weights[j][i] += 0.1 * a;
            }
        }
    }
    
    public void randomWeights() {
        Random ran = new Random();
        for(int i = 0; i < nodes.length; i++) {
            for(int j = 0; j < inputs.length; j++) {
                weights[i][j] = (ran.nextDouble() * 2);
            }
        }
    }
    
    public Node[] getNodes() {
        return nodes;
    }
    
    public double getWeight(Node j, Node k) {
        for(int i = 0; i < inputs.length; i++) {
            if(inputs[i] == j) {
                for(int r = 0; r < nodes.length; r++) {
                    if(nodes[r] == k) {
                        return weights[r][i];
                    }
                }
            }
        }
        System.out.println("Something went wrong");
        return 0;
    }
    
    public double[] getOutputs() {
        return outputs;
    }
    
    public void setOutputs(double[] outs) {
        for(int i = 0; i < outs.length; i++) {
            nodes[i].setOut(outs[i]);
        }
        outputs = outs;
    }
}
