package ml.app;

import ml.linear.LinearRegression;
import ml.knn.KNNRegression;
import ml.knn.KNNClassification;
import ml.model_selection.DataUtils;
import ml.preprocessing.MinMaxScaler;
import ml.preprocessing.StandardScaler;

// Classe pilotant les tests du framework.
public class Main {
    public static void main(String[] args) {
        // En-tête de la console.
        System.out.println("============================================================");
        System.out.println("    MINI SCIKIT-LEARN - EXPÉRIMENTATIONS COMPLÈTES");
        System.out.println("============================================================");

        //DATASETS
        
        // Dataset de régression (y ≈ 2x).
        double[][] regData = {
            {1.0, 2.1}, {2.0, 3.9}, {3.0, 6.1}, {4.0, 7.9}, {5.0, 10.2},
            {6.0, 11.8}, {7.0, 14.1}, {8.0, 15.9}, {9.0, 18.0}, {10.0, 20.1}
        };

        // Dataset de classification binaire.
        double[][] classData = {
            {1.0, 1.1, 0.0}, {1.2, 0.9, 0.0}, {0.8, 1.2, 0.0},
            {1.5, 1.0, 0.0}, {0.9, 1.3, 0.0},
            {5.0, 5.1, 1.0}, {5.2, 4.9, 1.0}, {4.8, 5.2, 1.0},
            {5.1, 5.0, 1.0}, {4.9, 5.3, 1.0}
        };

        //EXPÉRIENCE 1 : IMPACT DU LEARNING RATE
        System.out.println("\n============================================================");
        System.out.println("EXPÉRIENCE 1 : Impact du Learning Rate");
        System.out.println("============================================================");

        double[] learningRates = {0.01, 0.005, 0.001};
        for (double lr : learningRates) {
            System.out.println("\n--- Configuration : LR = " + lr + " ---");
            DataUtils.SplitResult split = DataUtils.trainTestSplit(regData, 0.2, 42);
            LinearRegression model = new LinearRegression(lr, 1000);

            model.printStatus(); 
            model.train(split.trainSet); 
            double s = model.score(split.testSet); 
            System.out.println("Score (R2) du modèle LinearRegression = " + s);
        }

        //EXPÉRIENCE 1B : IMPACT DU NOMBRE D'EPOCHS
        System.out.println("\n============================================================");
        System.out.println("EXPÉRIENCE 1B : Impact du Nombre d'Epochs");
        System.out.println("============================================================");

        int[] epochsValues = {500, 1000, 2000};
        for (int epochs : epochsValues) {
            System.out.println("\n--- Configuration : Epochs = " + epochs + " ---");
            DataUtils.SplitResult split = DataUtils.trainTestSplit(regData, 0.2, 42);
            LinearRegression model = new LinearRegression(0.01, epochs);

            model.printStatus();
            model.train(split.trainSet);
            double s = model.score(split.testSet);
            System.out.println("Score (R2) du modèle LinearRegression = " + s);
        }

        //EXPÉRIENCE 2 : IMPACT DE K DANS KNN RÉGRESSION
        System.out.println("\n============================================================");
        System.out.println("EXPÉRIENCE 2 : Impact de K dans KNN Régression");
        System.out.println("============================================================");
        int[] kValues = {1, 3, 5, 7};
        for (int k : kValues) {
            DataUtils.SplitResult split = DataUtils.trainTestSplit(regData, 0.2, 42);
            KNNRegression model = new KNNRegression(k);

            model.printStatus();
            model.train(split.trainSet);
            double s = model.score(split.testSet);
            System.out.println("Score (R2) du modèle " + model.getClass().getSimpleName() + " = " + s);
        }

        //EXPÉRIENCE 3 : IMPACT DU PRÉTRAITEMENT
        System.out.println("\n============================================================");
        System.out.println("EXPÉRIENCE 3 : Impact du Prétraitement");
        System.out.println("============================================================");

        // Données brutes.
        DataUtils.SplitResult splitNoPre = DataUtils.trainTestSplit(regData, 0.2, 42);
        KNNRegression knnNoPre = new KNNRegression(3);
        knnNoPre.printStatus();
        knnNoPre.train(splitNoPre.trainSet);
        System.out.println("Sans prétraitement -> Score (R2) du modèle = " + knnNoPre.score(splitNoPre.testSet));

        // Avec MinMaxScaler.
        DataUtils.SplitResult splitMinMax = DataUtils.trainTestSplit(regData, 0.2, 42);
        MinMaxScaler minmax = new MinMaxScaler();
        splitMinMax.trainSet = minmax.fitTransform(splitMinMax.trainSet);
        splitMinMax.testSet = minmax.transform(splitMinMax.testSet);
        KNNRegression knnMinMax = new KNNRegression(3);
        knnMinMax.printStatus();
        knnMinMax.train(splitMinMax.trainSet);
        System.out.println("Avec MinMaxScaler -> Score (R2) du modèle = " + knnMinMax.score(splitMinMax.testSet));

        // Avec StandardScaler.
        DataUtils.SplitResult splitStd = DataUtils.trainTestSplit(regData, 0.2, 42);
        StandardScaler std = new StandardScaler();
        splitStd.trainSet = std.fitTransform(splitStd.trainSet);
        splitStd.testSet = std.transform(splitStd.testSet);
        KNNRegression knnStd = new KNNRegression(3);
        knnStd.printStatus();
        knnStd.train(splitStd.trainSet);
        System.out.println("Avec StandardScaler -> Score (R2) du modèle = " + knnStd.score(splitStd.testSet));

        //EXPÉRIENCE 4 : IMPACT DU TEST RATIO
        System.out.println("\n============================================================");
        System.out.println("EXPÉRIENCE 4 : Impact du Test Ratio");
        System.out.println("============================================================");

        double[] testRatios = {0.2, 0.3, 0.4};
        for (double ratio : testRatios) {
            DataUtils.SplitResult split = DataUtils.trainTestSplit(regData, ratio, 42);
            KNNRegression model = new KNNRegression(3);
            model.printStatus();
            model.train(split.trainSet);
            double s = model.score(split.testSet);
            System.out.printf("Test Ratio = %.1f -> Score (R2) = %.4f (train=%d, test=%d)%n",
                    ratio, s, split.trainSet.length, split.testSet.length);
        }

        //EXPÉRIENCE 5 : KNN CLASSIFICATION
        System.out.println("\n============================================================");
        System.out.println("EXPÉRIENCE 5 : KNN Classification");
        System.out.println("============================================================");

        int[] kValuesClass = {1, 3, 5, 7};
        for (int k : kValuesClass) {
            DataUtils.SplitResult split = DataUtils.trainTestSplit(classData, 0.2, 42);
            KNNClassification model = new KNNClassification(k);

            model.printStatus();
            model.train(split.trainSet);
            double s = model.score(split.testSet);
            System.out.println("Score (accuracy) du modèle KNNClassification = " + s);
        }

        // Fin des tests.
        System.out.println("\n============================================================");
        System.out.println("FIN DES EXPÉRIMENTATIONS");
        System.out.println("============================================================");
    }
}