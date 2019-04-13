/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnet2;

import java.io.*;
import static java.lang.Math.abs;
import static java.lang.Math.exp;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

/**
 *
 * @author James
 */
public class NeuralNet2 {

    /**
     * @param args the command line arguments
     */
    static int dataSize;
    static int testSize;
    static int numFeatures;
    public static void main(String[] args) throws IOException {
        int numLayers = 0;
        int hNodes = 200;
        int iNodes = 0;
        int oNodes = 0;
        int func;
        int netType;
        int sigType;
        int pop;
        int alg;
        double[][] in;
        double[] ex;
        String train;
        String test;
        //First gets input from user about network parameters
        Scanner scan = new Scanner(System.in);
        System.out.print("What Dataset should be used? ");
        train = scan.nextLine();
        System.out.print("How large is this Dataset? ");
        dataSize = scan.nextInt();
        System.out.println("What test set should be used? ");
        test = scan.nextLine();
        test = scan.nextLine();
        System.out.print("How large is the test set? ");
        testSize = scan.nextInt();
        System.out.print("How many input nodes should there be? ");
        iNodes = scan.nextInt();
        System.out.print("How many output nodes should there be? ");
        oNodes = scan.nextInt();
        System.out.print("How big should the population be? ");
        pop = scan.nextInt();
        System.out.print("What algorithm: Backprop (1), Genetic (2), Evolution Strategy (3), or Differential Evolution (4)? ");
        alg = scan.nextInt();
        numFeatures = (iNodes * hNodes) + (hNodes * oNodes);
        
        //Initialize networks
        Network[] networks = new Network[pop];
        for(int i = 0; i < pop; i++){
            networks[i] = new Network(3, hNodes, iNodes, oNodes);
        }
        //Initialize population representation, check if running Evolution Strategy to determine whether or not to generate sigmas
        double[][] population;
        if(alg == 3) {
            population = new double[pop][numFeatures*2];
            for(int i = 0; i < pop; i++) {
                double[] a = networks[i].getRepresentaion();
                for(int j = 0; j < numFeatures; j++) {
                    population[i][j] = a[j];
                }
                Random ran = new Random();
                for(int j = 0; j < numFeatures-1; j++) {
                    population[i][numFeatures+j] = abs(ran.nextGaussian());
                }
            }
        } else {
            population = new double[pop][numFeatures];
            for(int i = 0; i < pop; i++) {
                population[i] = networks[i].getRepresentaion();
            }
        }
        //Initialize training set of inputs and expected outputs
        in = new double[dataSize][iNodes];
        ex = new double[dataSize];
        readIn(train, in, ex);
        //Then initialize testing set of inputs and expected outputs
        double[][] inTest = new double[dataSize][iNodes];
        double[] exTest = new double[dataSize];
        readIn(test, inTest, exTest);
        for(int f = 0; f < 20; f++) {
            for(int j = 0; j < dataSize; j++) {
                //Run networks on training set
                for(int i = 0; i < pop; i++) {
                    networks[i].activate(in[j], ex[j]);
                }
                //backpropagate if neccesary
                if(alg == 1) {
                    for(int i = numLayers-2; i > 0; i--) {
                        networks[i].backprop(ex[j]);
                    }
                }
            }
            //Reinitialize population, again checking if running Evolution Strategy
            if(alg == 3) {
                population = new double[pop][numFeatures*2];
                for(int i = 0; i < pop; i++) {
                    double[] a = networks[i].getRepresentaion();
                    for(int j = 0; j < numFeatures; j++) {
                        population[i][j] = a[j];
                    }
                    Random ran = new Random();
                    for(int j = 0; j < numFeatures-1; j++) {
                        population[i][numFeatures+j] = abs(ran.nextGaussian());
                    }
                }
            } else {
                population = new double[pop][numFeatures];
                for(int i = 0; i < pop; i++) {
                    population[i] = networks[i].getRepresentaion();
                }
            }
            //Get fitness of all networks
            double[] fit = fitness(networks, dataSize);
            Random ran = new Random();
            //Run algorithm that was selected earlier
            switch(alg) {
                case 2: // Genetic
                    int best = 0;
                    int second = 0;
                    for(int i = 0; i < fit.length; i++) { //Find the top two individuals
                        if(fit[i] < fit[best]) {
                            second = best;
                            best = i;
                        } else if (fit[i] < fit[second]) {
                            second = i;
                        }
                    }
                    //Use the top two individuals to generate children via crossover
                    double parentsGA[][] = new double[2][numFeatures];
                    parentsGA[0] = population[best];
                    parentsGA[1] = population[second];
                    double[][] childrenGA = crossover(alg, parentsGA, 2);
                    mutation(alg, childrenGA); //mutate those children
                    Network[] cNetworksGA = new Network[2];
                    cNetworksGA[0] = new Network(3, hNodes, iNodes, oNodes, childrenGA[0]);
                    cNetworksGA[1] = new Network(3, hNodes, iNodes, oNodes, childrenGA[1]); //Create networks from those children
                    for(int j = 0; j < dataSize; j++) {
                        //Run network on input
                        for(int i = 0; i < 2; i++) {
                            cNetworksGA[i].activate(in[j], ex[j]); //Run the training set on those children
                        }
                    }
                    double[] cFitGA = fitness(cNetworksGA, dataSize); //Get the fitness of the children
                    for(int i = 0; i < 2; i++) {
                        int temp = 0;
                        for(int j = 0; j < fit.length; j++) {
                            if(fit[j] > fit[temp]) {
                                temp = j;
                            }
                        }
                        /*For each child representation compare that child to the least fit
                        individual and relace the worst with the best*/
                        if(fit[temp] >  cFitGA[i]) {
                            population[temp] = childrenGA[i];
                            fit[temp] = cFitGA[i];
                            networks[temp] = cNetworksGA[i];
                        }
                    }
                    break;
                case 3: // ES
                    //Randomly select parents
                    double parents[][] = new double[2][numFeatures*2];
                    parents[0] = population[ran.nextInt(fit.length)];
                    parents[1] = population[ran.nextInt(fit.length)];
                    //Generate Children via Crossover, the mutate the children
                    double children[][] = crossover(alg, parents, 4);
                    mutation(alg, children);
                    //Put the child representations into networks and run them in the training
                    Network[] cNetworks = new Network[4];
                    cNetworks[0] = new Network(3, hNodes, iNodes, oNodes, children[0]);
                    cNetworks[1] = new Network(3, hNodes, iNodes, oNodes, children[1]);
                    cNetworks[2] = new Network(3, hNodes, iNodes, oNodes, children[2]);
                    cNetworks[3] = new Network(3, hNodes, iNodes, oNodes, children[3]);
                    for(int j = 0; j < dataSize; j++) {
                        //Run network on input
                        for(int i = 0; i < 4; i++) {
                            cNetworks[i].activate(in[j], ex[j]);
                        }
                    }
                    //Get the fitness of the child networks
                    double[] cFit = fitness(cNetworks, dataSize);
                    /*For each child representation compare that child to the least fit
                    individual and relace the worst with the best*/
                    for(int i = 0; i < 4; i++) {
                        int temp = 0;
                        for(int j = 0; j < fit.length; j++) {
                            if(fit[j] > fit[temp]) {
                                temp = j;
                            }
                        }
                        /*If the worst fitness individual currently in the population is worse that the child
                        replace that individual with the child*/
                        if(fit[temp] >  cFit[i]) {
                            population[temp] = children[i];
                            fit[temp] = cFit[i];
                            networks[temp] = cNetworks[i];
                        }
                    }
                    break;
                case 4: // DE
                    double low = fit[0]; //set initial value to compare fitness
                    int lowIndex = 0; //keep track of index for fittest individual
                    for(int i=1; i < fit.length; i++){ //go through remaining entries and compare
                        if(fit[i]<low){ //individual at index i is fitter than previous best
                            low=fit[i]; //store fitness value of best individual
                            lowIndex = i; //store index of best individual
                        }
                    }
                    
                    int xTarget = lowIndex; //Set the best individual as the target
                    int x1 = ran.nextInt(population.length-1); //Find a random new individual other that the target
                    while (xTarget==x1){
                        x1 = ran.nextInt(population.length-1); //If the index for x1 is the same as index for x1, generate until they are different
                    }
                    int x2 = ran.nextInt(population.length-1); //Find a second new individual
                    while (xTarget==x2 || x1==x2){
                        x2 = ran.nextInt(population.length-1); //Ensures unique individuals for xTarget, x1, and x2
                    }
                    int x3 = ran.nextInt(population.length-1);//Find a third new individual
                    while(xTarget==x3 || x1==x3 || x2==x3){
                        x3 = ran.nextInt(population.length-1); //Ensures unique individuals for xTarget, x1, x2, x3
                    }

                    double xDiff[] = new double[population[0].length]; //Create a new vector to hold the difference between x2 and x3
                    for(int i = 0; i<population[0].length; i++){
                            xDiff[i] = population[x2][i] - population[x3][i]; //Calculate the difference vector from the second and third random individuals
                    }

                    double xTrial[] = new double[population[0].length]; //Create a vector to hold the trial vector
                    for(int i = 0; i<population[0].length; i++){
                            xTrial[i] = population[x1][i] + (0.5 * xDiff[i]); //Calculate the trial vector from individual x1 and the difference vector
                    }

                    double parentsDE[][] = new double[2][population[0].length]; //Create a parents array using
                    parentsDE[0]= population[xTarget]; //The best individual
                    parentsDE[1]= xTrial; //and the trial vector

                    double[][] childrenDE = crossover(4, parentsDE, 1); //Create an array to hold offspring and use crossover to create the offsping itself
                    Network[] cNetwork = {new Network(3, hNodes, iNodes, oNodes, childrenDE[0])}; //Creates a new network using the representation of the offspring
                    for(int j = 0; j < dataSize; j++) {
                        //Run network on input
                        cNetwork[0].activate(in[j], ex[j]);
                    }
                    double[] dFit = fitness(cNetwork, dataSize); //Calculate the fitness of the offspring
                    if(dFit[0] < low) { //If the offspring has a lower fitness(error) than the target vector (fittest individual):
                        population[lowIndex] = childrenDE[0]; //Replace the target vector with the offspring in the population of representations
                        networks[lowIndex] = cNetwork[0]; //and replace it in the array of networks
                    }
                    break;

            }
            System.out.println("Generation: " + f);
            for(int h = 0; h < networks.length; h++) {
                networks[h].error = 0;
            }
        }
        double[] fit = fitness(networks, dataSize);
        double low = fit[0]; //set initial value to compare fitness
        int lowIndex = 0; //keep track of index for fittest individual
        for(int i=1; i < fit.length; i++){ //go through remaining entries and compare
            if(fit[i]<low){ //individual at index i is fitter than previous best
                low=fit[i]; //store fitness value of best individual
                lowIndex = i; //store index of best individual
            }
        }
        //Find the fittest network and run the test set on it
        Network[] fittest = new Network[1];
        fittest[0] = new Network(3, hNodes, iNodes, oNodes, population[lowIndex]);
        for(int t = 0; t < testSize; t++) {
            fittest[0].activate(inTest[t], exTest[t]);
        }
        //Find and print the fitness on the test set
        double[] testFit = fitness(fittest, testSize);
        System.out.println(testFit[0]);
    }
    //This method reads input from a file
    public static void readIn(String data, double[][] in, double[] ex) throws FileNotFoundException, IOException {
        String[] split = new String[in[0].length];
        List<String> temp = new ArrayList<String>();
        FileReader filereader = new FileReader(data);
        BufferedReader reader = new BufferedReader(filereader);
        String line = null;
        while((line = reader.readLine()) != null) {
            temp.add(line);
        }
        for(int i = 0; i < temp.size(); i++) {
            split = temp.get(i).split(",");
            for(int j = 0; j < split.length - 1; j++) {
                in[i][j] = Double.parseDouble(split[j]);
            }
            ex[i] = Double.parseDouble(split[split.length - 1]);
        }
    }
    //Method to find the root squared error, used to measure fitness
    public static double[] fitness(Network[] networks, int size) {
        double[] fit = new double[networks.length];
        for(int i = 0; i < fit.length; i++) {
            fit[i] = sqrt(networks[i].error / size);
        }
        return fit;
    }
    
    public static double[][] crossover(int alg, double[][] parents, int numChildren) {
        double[][] children = null;
        Random ran = new Random();
        switch(alg) {
            case 2: // Genetic
                //Create an array to hold the children
                children = new double[numChildren][numFeatures];
                for(int i = 0; i < numChildren; i++) {
                    int swapPoint1 = ran.nextInt(numFeatures-1); //Find two random swap points
                    int swapPoint2 = numFeatures - ran.nextInt(numFeatures-swapPoint1);
                    for (int geneIndex = 0; geneIndex < numFeatures; geneIndex++) {
                        //Determine which parents genes to use based on swap points
                        if (geneIndex < swapPoint1 || geneIndex > swapPoint2) {
                                children[i][geneIndex] = parents[0][geneIndex];
                        } else {
                                children[i][geneIndex] = parents[1][geneIndex];
                        }
                    }
                }
                break;
            case 3: // ES
                //Create an array to hold the children
                children = new double[numChildren][numFeatures*2];
                double temp;
                for(int i = 0; i < numChildren; i++) {
                    for(int j = 0; j < numFeatures-1; j++) {
                        //For each gene there is a 50/50 chance of the child inheriting that gene from either parent
                        temp = ran.nextDouble();
                        if (temp < 0.5) {
                            children[i][j] = parents[1][j];
                            children[i][j+numFeatures] = parents[1][j+numFeatures];
                        } else {
                            children[i][j] = parents[0][j];
                            children[i][j+numFeatures] = parents[0][j+numFeatures];
                        }
                    }
                }
                break;
            case 4: // DE
                children = new double[numChildren][parents[0].length]; //Creates an array of vectors for the offspring
                for (int i = 0; i < parents[0].length; i++){ //For each gene in the chromosome
                    int c = ran.nextInt(2); //calculate an integer: either 0 or 1
                    if (c == 1){ //if the number generated is 1, the gene for the offspring comes from the trial vector
                        children[0][i]=parents[1][i];
                    }
                    else{ //The number generated is 0 and the gen from the target vector is passed to the offspring
                        children[0][i]=parents[0][i];
                    }
                }
                break;
        }
        //return the generated children
        return children;
    }
    
    public static void mutation(int alg, double[][] parents) {
        Random ran = new Random();
        switch(alg) {
            case 2: // Genetic
                double mutationRate = 0.015; //Chance that any particular gene will mutate
                double min = -0.1;
                double max = 0.1;
                for(int i = 0; i < parents.length; i++) {
                    for(int j = 0; j < parents[0].length; j++) {
                        if(ran.nextDouble() < mutationRate) {
                            parents[i][j] = parents[i][j] + (min + (max - min) * ran.nextDouble()); //Mutate by a random number between -0.1 and 0.1
                        }
                    }
                }
                break;
            case 3: // ES
                for(int i = 0; i < parents.length; i++) {
                    for(int j = 0; j < numFeatures; j++) {
                        //Mutate X
                        parents[i][j] = parents[i][j] + (parents[i][j+numFeatures] * abs(ran.nextGaussian()));
                        //Run Evolution Streategy on Sigma
                        parents[i][j+numFeatures] = parents[i][j+numFeatures] * exp((1 / sqrt(numFeatures) ) * abs(ran.nextGaussian()));
                    }
                }
                break;
            case 4: // DE
                
                break;
        }
    }
}
