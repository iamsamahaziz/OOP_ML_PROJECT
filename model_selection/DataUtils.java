package ml.model_selection;

import java.util.Random;

// Utilitaire pour la gestion et la préparation des données.
public class DataUtils {
    
    // Conteneur pour les sous-ensembles d'entraînement et de test.
    public static class SplitResult { 
        public double[][] trainSet; // Données d'apprentissage.
        public double[][] testSet;  // Données d'évaluation.
    }

    // Divise un dataset de manière aléatoire et reproductible.
    public static SplitResult trainTestSplit(double[][] dataset, double testRatio, long seed) {
        
        // Vérification de sécurité.
        if (dataset == null || dataset.length == 0) {
            throw new IllegalArgumentException("Le dataset ne peut pas être vide.");
        }

        int n = dataset.length;

        // Création d'un tableau d'indices pour le mélange.
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }

        // Mélange de Fisher-Yates avec graine aléatoire.
        Random rand = new Random(seed);
        for (int i = n - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        // Calcul des dimensions des sous-ensembles.
        int testSize = (int) (testRatio * n);
        int trainSize = n - testSize;

        // Répartition des données selon les indices mélangés.
        SplitResult res = new SplitResult();
        res.trainSet = new double[trainSize][];
        res.testSet = new double[testSize][];

        for (int i = 0; i < trainSize; i++) {
            res.trainSet[i] = dataset[indices[i]];
        }

        for (int i = 0; i < testSize; i++) {
            res.testSet[i] = dataset[indices[trainSize + i]];
        }

        return res; 
    }
}