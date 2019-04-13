/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnet;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import static java.lang.Math.pow;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

/**
 *
 * @author James
 */
public class NeuralNet {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        int hLayers;
        int sigType = 0;
        int inputs;
        int outputs;
        int nodes = 0;
        int type;
        
        Scanner scan = new Scanner(System.in);
        System.out.print("Should the network be a Feed-Forward Net (1), or a Radial Basis Function (2) ");
        type = scan.nextInt();
        if (type == 2) {
            hLayers = 1;
            System.out.print("One sigma (1), two sigmas (2), or random(3)? ");
            sigType = scan.nextInt();
        } else {
            System.out.print("How many hidden layers should there be? ");
            hLayers = scan.nextInt();
        }
        if(hLayers > 0) {
            System.out.print("How many nodes in each hidden layer should there be? ");
            nodes = scan.nextInt();
        }
        System.out.print("How many input nodes should there be? ");
        inputs = scan.nextInt();
        System.out.print("How many output nodes should there be? ");
        outputs = scan.nextInt();
        
        
        Layer input = new Layer(inputs);
        Layer output = null;
        Layer[] hidden = null;
        if(hLayers > 0) {
            hidden = new Layer[hLayers];
            hidden[0] = new Layer(input.getNodes(), nodes, type, input);
            for(int i = 1; i < hLayers; i++) {
                hidden[i] = new Layer(hidden[i-1].getNodes(), nodes, type, hidden[i-1]);
            }
            output = new Layer(hidden[hLayers-1].getNodes(), outputs, 3, hidden[hLayers-1]);
            if(sigType != 0) {
                for(int i = 0; i < hLayers; i++) {
                    hidden[i].setSigmas(sigType);
                }
            }
        } else {
            output = new Layer(input.getNodes(), outputs, 3, input);
        }
        
        double[][] in = null;
        double[] e = null;
        double[] o;
        int size = 0;
        String f;
        switch(inputs) {
            case 2:
                f = ("C:\\Users\\James\\Documents\\NetBeansProjects\\NeuralNet\\src\\neuralnet\\2.txt");
                size = 200;
                in = new double[size][inputs];
                e = new double[200];
                pop(in);
                rose(in, e);
                break;
            case 3:
                f = ("3");
                size = 400;
                in = new double[size][inputs];
                e = new double[400];
                pop(in);
                rose(in, e);
                break;
            case 4:
                f = ("4");
                size = 750;
                in = new double[size][inputs];
                e = new double[750];
                pop(in);
                rose(in, e);
                break;
            case 5:
                f = ("5");
                size = 1500;
                in = new double[size][inputs];
                e = new double[1500];
                pop(in);
                rose(in, e);
                break;
            case 6:
                f = ("6");
                size = 3000;
                in = new double[size][inputs];
                e = new double[3000];
                pop(in);
                rose(in, e);
                break;
        }
        int c = 0;
        int cb = 0;
        double msq = 0;
        double sq = 0;
        double[] ex = new double[outputs];
        for(int g = 0; g < 9001; g++) {
            for(int i = 0; i < in.length; i++) {
                input.setOutputs(in[0]);
                for(int j = 0; j < hLayers; j++) {
                    hidden[j].activate();
                }
                output.activate();
                for(int j = 0; j < outputs; j++) {
                    ex[j] = e[0];
                }
                sq = pow(output.nodes[0].getOutput() - ex[0], 2);
                if (sq < 100000) {
                    cb++;
                } else {
                    cb = 0;
                }
                if(type == 1) {
                    o = output.getOutputs();
                    output.backpropO(ex);
                    for(int d = hLayers-1; d >= 0; d--) {
                        if(d == (hLayers-1)) {
                            hidden[d].backpropH(output);
                        } else {
                            hidden[d].backpropH(hidden[d+1]);
                        }
                    }
                }
                System.out.println(output.outputs[0] + " " + e[0]);
//                  System.out.println(output.weights[0][0]);
            }
            if(cb == size) {
                c++;
            }
            if(c ==20) {
                System.out.println("Convergence Rate" + g);
                break;
            }
        }
    }

    public static void getIn(String filename, double[][] in, double[] out) {
      List<String> records = new ArrayList<String>();
      try
      {
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line;
        while ((line = reader.readLine()) != null)
        {
          records.add(line);
        }
        reader.close();
      }
      catch (Exception e)
      {
        System.err.format("Exception occurred trying to read '%s'.", filename);
        e.printStackTrace();
      }
        int k = 0;
        int h = 0;
        for(int i = 0; i < in.length; i++) {
            String a = records.get(i);
            a = a.replaceAll("\\[", "").replaceAll("\\]","");
            if((i % 2) == 0) {
                for(int j = 0; j < in[i].length; j++) {
                    String[] b = a.split(", ");
                    in[k][j] = Integer.parseInt(b[j]);
                }
                k++;
            } else {
                out[h] = Double.parseDouble(a);
                h++;
            }
        }
    }
    public static void rose(double[][] in, double[] out) {
        for(int i = 0; i < in.length; i++) {
            for(int j = 0; j < in[i].length-1; j++) {
                out[i] += (pow((1 - in[i][j]), 2) + ((100 * pow(in[i][j+1] - (in[i][j] * in[i][j]), 2))));
            }
        }
    }
    public static void pop(double[][] in) {
        Random ran = new Random();
        for(int i = 0; i < in.length; i++) {
            for(int j = 0; j < in[i].length; j++) {
                in[i][j] = ran.nextInt(50);
            }
        }
    }
}