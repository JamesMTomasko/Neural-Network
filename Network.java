/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnet2;

import static java.lang.Math.pow;

/**
 *
 * @author James
 */
public class Network {
    double representation[];
    int numLayers;
    Layer[] layers;
    int hNodes;
    int iNodes;
    int oNodes;
    double error = 0;
    //Constructor without provided representaion
    public Network(int numLayers, int hNodes, int iNodes, int oNodes) {
        //Initialize variables
        this.numLayers = numLayers;
        this.hNodes = hNodes;
        this.iNodes = iNodes;
        this.oNodes = oNodes;
        //Initialize all layers
        layers = new Layer[3];
        layers[0] = new Layer(1, iNodes, 2, 0);
        if(numLayers > 0) {
            for(int i = 1; i < numLayers - 1; i++){
                layers[i] = new Layer(2, hNodes, 2, layers[i-1].nodes.length);
            }
        }
        layers[numLayers-1] = new Layer(3, oNodes, 2, layers[numLayers-2].nodes.length);
        //Initialize representation
        representation = new double[(iNodes * hNodes) + (hNodes * oNodes)];
        createRepresentation();
    }
    //Constructor with provided representation
    public Network(int numLayers, int hNodes, int iNodes, int oNodes, double[] representation) {
        //Initialize variables
        this.numLayers = numLayers;
        this.hNodes = hNodes;
        this.iNodes = iNodes;
        this.oNodes = oNodes;
        //Initialize all layers
        layers = new Layer[3];
        layers[0] = new Layer(1, iNodes, 1, 0);
        if(numLayers > 0) {
            for(int i = 1; i < numLayers - 1; i++){
                layers[i] = new Layer(2, hNodes, 2, layers[i-1].nodes.length);
            }
        }
        layers[numLayers-1] = new Layer(3, oNodes, 2, layers[numLayers-2].nodes.length);
        //Initialize representation and use that to set weights
        this.representation = representation;
        int i = 0;
        for(int j = 1; j < numLayers; j++) {
            Layer temp = layers[j];
            for(int k = 0; k < temp.nodes.length; k++) {
                for(int l = 0; l < temp.weights[0].length; l++) {
                    temp.weights[k][l] = this.representation[i];
                    i++;
                }
            }
        }
    }
    //Generates representaion from weights
    public void createRepresentation() {
        Layer temp;
        int i = 0;
        for(int j = 1; j < numLayers; j++) {
            temp = layers[j];
            for(int k = 0; k < temp.nodes.length; k++) {
                for(int l = 0; l < temp.weights[0].length; l++) {
                    representation[i] = temp.weights[k][l];
                    i++;
                }
            }
        }
    }
    
    public double[] getRepresentaion() {
        return representation;
    }
    //Method to run network on a given input
    public void activate(double[] in, double ex) {
        double temp = 0;
        layers[0].outputs = in;
        for(int i = 1; i < numLayers; i++) {
            layers[i].activate(layers[i-1].outputs);
        }
        temp = layers[2].outputs[0] - ex; //Find error
        error += pow(temp, 2); //Add to running total of mean squared error
    }
    //Backpropagation function
    public void backprop(double ex) {
        layers[numLayers-1].backpropO(ex, layers[numLayers-2].outputs);
        for(int i = numLayers-2; i > 0; i--) {
            layers[i].backpropH(layers[i+1], layers[i-1].outputs);
        }
    }
}
