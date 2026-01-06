package ml.knn;

import ml.core.MLModel;
import ml.metrics.Metrics;
import java.util.*;

// K-Plus Proches Voisins (KNN) pour la Classification.
public class KNNClassification extends MLModel {
    private int k;
    private double[][] trainingData;

    public KNNClassification(int k) {
        super("KNN Classification (k=" + k + ")");
        if (k <= 0) k = 1;
        this.k = k;
    }

    @Override
    public void train(double[][] dataset) {
        this.trainingData = dataset;
    }

    private double euclideanDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }

    @Override
    public double predict(double[] input) {
        // 1. Calculer distances
        List<double[]> neighbors = new ArrayList<>();

        for (double[] row : trainingData) {
            double[] features = new double[row.length - 1];
            for (int i = 0; i < row.length - 1; i++) {
                features[i] = row[i];
            }
            double distance = euclideanDistance(input, features);
            double label = row[row.length - 1];
            neighbors.add(new double[]{distance, label});
        }

        // 2. Trier par distance
        neighbors.sort((a, b) -> Double.compare(a[0], b[0]));
        int actualK = Math.min(k, neighbors.size());

        // 3. Récupérer les labels des k voisins
        double[] labels = new double[actualK];
        for (int i = 0; i < actualK; i++) {
            labels[i] = neighbors.get(i)[1];
        }

        // 4. Trouver toutes les classes uniques
        List<Double> classesUniques = new ArrayList<>();
        for (int i = 0; i < labels.length; i++) {
            boolean existe = false;
            for (int j = 0; j < classesUniques.size(); j++) {
                if (classesUniques.get(j) == labels[i]) {
                    existe = true;
                    break;
                }
            }
            if (!existe) {
                classesUniques.add(labels[i]);
            }
        }

        // 5. Compter les votes pour chaque classe
        double meilleurClasse = classesUniques.get(0);
        int maxVotes = 0;

        for (int i = 0; i < classesUniques.size(); i++) {
            double classe = classesUniques.get(i);
            int votes = 0;
            
            for (int j = 0; j < labels.length; j++) {
                if (labels[j] == classe) {
                    votes++;
                }
            }

            // Mettre à jour si plus de votes ou égalité avec classe plus petite
            if (votes > maxVotes || (votes == maxVotes && classe < meilleurClasse)) {
                meilleurClasse = classe;
                maxVotes = votes;
            }
        }

        return meilleurClasse;
    }

    @Override
    public double score(double[][] testSet) {
        double[] yTrue = new double[testSet.length];
        double[] yPred = new double[testSet.length];

        for (int i = 0; i < testSet.length; i++) {
            yTrue[i] = testSet[i][testSet[i].length - 1];

            double[] features = new double[testSet[i].length - 1];
            for (int j = 0; j < testSet[i].length - 1; j++) {
                features[j] = testSet[i][j];
            }

            yPred[i] = predict(features);
        }

        return Metrics.accuracy(yTrue, yPred);
    }
}