package ml.knn;

import ml.core.MLModel;
import ml.metrics.Metrics;
import java.util.Arrays;

// Algorithme des K-Plus Proches Voisins (KNN) pour la Régression.
public class KNNRegression extends MLModel {
    private int k; // Nombre de voisins.
    private double[][] trainingData; // Données d'entraînement en mémoire.

    // Initialise le modèle avec k voisins.
    public KNNRegression(int k) {
        super("KNN Regression (k=" + k + ")");
        if (k <= 0) {
            throw new IllegalArgumentException("k doit être strictement positif !");
        }
        this.k = k;
    }

    // Apprentissage passif : mémorisation du dataset.
    @Override
    public void train(double[][] dataset) {
        this.trainingData = dataset;
    }

    // Calcule la distance euclidienne entre deux points.
    private double euclideanDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    // Extrait les caractéristiques (exclut la cible).
    private double[] extractFeatures(double[] row) {
        int numFeatures = row.length - 1; 
        double[] features = new double[numFeatures];
        
        for (int i = 0; i < numFeatures; i++) {
            features[i] = row[i];
        }
        
        return features;
    }

    // Prédit la moyenne des valeurs cibles des k plus proches voisins.
    @Override
    public double predict(double[] input) {
        double[][] distances = new double[trainingData.length][2];

        for (int i = 0; i < trainingData.length; i++) {
            double[] features = extractFeatures(trainingData[i]);
            distances[i][0] = euclideanDistance(input, features);
            distances[i][1] = trainingData[i][trainingData[i].length - 1];
        }

        // Tri par distance croissante.
        Arrays.sort(distances, (a, b) -> Double.compare(a[0], b[0]));

        int actualK = Math.min(k, distances.length);

        // Moyenne des cibles des voisins.
        double sumY = 0;
        for (int i = 0; i < actualK; i++) {
            sumY += distances[i][1];
        }
        
        return sumY / actualK;
    }

    // Calcule le score R² sur le jeu de test.
    @Override
    public double score(double[][] testSet) {
        double[] yTrue = new double[testSet.length];
        double[] yPred = new double[testSet.length];

        for (int i = 0; i < testSet.length; i++) {
            yTrue[i] = testSet[i][testSet[i].length - 1];
            double[] input = extractFeatures(testSet[i]);
            yPred[i] = predict(input);
        }
        
        return Metrics.r2Score(yTrue, yPred);
    }
}