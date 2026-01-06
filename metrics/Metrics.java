package ml.metrics;

// Fonctions pour calculer les scores de performance.
public class Metrics {
    
    // Calcule le score R2.
    // Permet de voir si le modèle explique bien les données.
    public static double r2Score(double[] yTrue, double[] yPred) {
        // Moyenne des vraies valeurs.
        double moyenne = 0;
        for (int i = 0; i < yTrue.length; i++) {
            moyenne += yTrue[i];
        }
        moyenne /= yTrue.length;

        // Somme des erreurs au carré.
        double erreur = 0;
        for (int i = 0; i < yTrue.length; i++) {
            erreur += Math.pow(yTrue[i] - yPred[i], 2);
        }

        // Somme des écarts totaux par rapport à la moyenne.
        double écartTotal = 0;
        for (int i = 0; i < yTrue.length; i++) {
            écartTotal += Math.pow(yTrue[i] - moyenne, 2);
        }
        
        // Calcul final du R2.
        return 1 - (erreur / écartTotal); 
    }

    // Calcule le taux de bonnes réponses (Accuracy).
    public static double accuracy(double[] yTrue, double[] yPred) {
        int nbCorrect = 0;
        for (int i = 0; i < yTrue.length; i++) {
            // On arrondit pour comparer les classes 0 ou 1.
            if (Math.round(yPred[i]) == Math.round(yTrue[i])) {
                nbCorrect++; 
            }
        }
        return (double) nbCorrect / yTrue.length; 
    }

    // Calcule l'erreur moyenne au carré (MSE).
    public static double mse(double[] yTrue, double[] yPred) {
        double sommeErreurs = 0.0;
        for (int i = 0; i < yTrue.length; i++) {
            sommeErreurs += Math.pow(yTrue[i] - yPred[i], 2);
        }
        return sommeErreurs / yTrue.length;
    }
}