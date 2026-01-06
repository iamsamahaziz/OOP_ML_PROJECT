package ml.linear;

import ml.core.MLModel;
import ml.metrics.Metrics;

// Régression Linéaire simple (1D) par descente de gradient.
public class LinearRegression extends MLModel {
    private double slope = 0.0;         // Pente m.
    private double intercept = 0.0;     // Biais b.
    private double learningRate = 0.01;  // Pas d'apprentissage.
    private int numEpochs = 1000;       // Nombre d'itérations.

    // Constructeur par défaut.
    public LinearRegression() {
        super("Linear Regression");
    }

    // Constructeur avec hyperparamètres personnalisés.
    public LinearRegression(double learningRate, int numEpochs) {
        super("Linear Regression");
        this.learningRate = learningRate;
        this.numEpochs = numEpochs;
    }

    // Entraîne le modèle sur le dataset.
    @Override
    public void train(double[][] dataset) {
        if (isDatasetValid(dataset)) {
            initializeParameters(); 
            gradientDescentLoop(dataset); 
        }
    }

    // Initialise les paramètres à zéro.
    private void initializeParameters() {
        this.slope = 0.0;
        this.intercept = 0.0;
    }

    // Vérifie la validité des données.
    private boolean isDatasetValid(double[][] dataset) {
        return dataset != null && dataset.length > 0;
    }

    // Optimisation des paramètres par itération.
    private void gradientDescentLoop(double[][] dataset) {
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            double[] gradients = computeGradients(dataset);
            updateParameters(gradients[0], gradients[1]);
            
            // Suivi de l'erreur tous les 200 cycles.
            if (epoch % 200 == 0) {
                double cost = computeCost(dataset);
                System.out.println("Epoch " + epoch + " - Coût (MSE): " + cost);
            }
        }
    }

    // Calcule les dérivées partielles (gradients).
    private double[] computeGradients(double[][] dataset) {
        double gradSlope = 0;
        double gradIntercept = 0;
        int n = dataset.length;

        for (double[] row : dataset) {
            double x = row[0];
            double yReal = row[row.length - 1];
            double yPred = slope * x + intercept; 
            
            gradSlope += (yPred - yReal) * x;
            gradIntercept += (yPred - yReal);
        }
        return new double[]{(2.0 / n) * gradSlope, (2.0 / n) * gradIntercept};
    }

    // Mise à jour des poids (pente et biais).
    private void updateParameters(double gradSlope, double gradIntercept) {
        slope -= learningRate * gradSlope;
        intercept -= learningRate * gradIntercept;
    }

    // Calcule le coût actuel (MSE).
    private double computeCost(double[][] dataset) {
        double cost = 0.0;
        int n = dataset.length;
        
        for (double[] row : dataset) {
            double x = row[0];
            double yTrue = row[row.length - 1];
            double yPred = slope * x + intercept;
            cost += Math.pow(yPred - yTrue, 2);
        }
        
        return cost / n;
    }

    // Prédit la valeur pour une entrée donnée.
    @Override
    public double predict(double[] input) {
        return slope * input[0] + intercept;
    }

    // Évalue le modèle avec le score R².
    @Override
    public double score(double[][] testSet) {
        double[] yTrue = new double[testSet.length];
        double[] yPred = new double[testSet.length];
        
        for (int i = 0; i < testSet.length; i++) {
            yTrue[i] = testSet[i][testSet[i].length - 1];
            yPred[i] = predict(new double[]{testSet[i][0]});
        }
        
        return Metrics.r2Score(yTrue, yPred);
    }
}