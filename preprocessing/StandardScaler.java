package ml.preprocessing;

// Standardisation Z-Score (moyenne 0, écart-type 1).
public class StandardScaler implements Preprocessor {
    private double[] mean;         // Moyennes.
    private double[] std;          // Écarts-types.
    private boolean fitted = false; // État d'entraînement.

    // Calcule moyenne et écart-type (ignore la cible y).
    @Override
    public void fit(double[][] dataset) {
        if (dataset == null || dataset.length == 0) {
            throw new IllegalArgumentException("Le dataset ne peut pas être vide.");
        }
        
        int n = dataset.length;             
        int p = dataset[0].length - 1;      

        mean = new double[p];
        std = new double[p];

        for (int j = 0; j < p; j++) {
            double somme = 0;
            for (int i = 0; i < n; i++) {
                somme += dataset[i][j];
            }
            mean[j] = somme / n;
        }

        for (int j = 0; j < p; j++) {
            double sommeCarres = 0;
            for (int i = 0; i < n; i++) {
                sommeCarres += Math.pow(dataset[i][j] - mean[j], 2);
            }
            std[j] = Math.sqrt(sommeCarres / n);
        }
        
        fitted = true; 
    }

    // Applique la transformation Z-Score.
    @Override
    public double[][] transform(double[][] dataset) {
        if (!fitted) {
            throw new IllegalStateException("fit() doit être appelé avant transform()");
        }
        
        int n = dataset.length;
        int p = dataset[0].length;
        double[][] result = new double[n][p];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p - 1; j++) {
                if (std[j] == 0) {  
                    result[i][j] = 0;
                } else {
                    result[i][j] = (dataset[i][j] - mean[j]) / std[j];
                }
            }
            result[i][p - 1] = dataset[i][p - 1];
        }
        return result;
    }

    // Entraîne et transforme en une étape.
    @Override
    public double[][] fitTransform(double[][] dataset) {
        fit(dataset);
        return transform(dataset);
    }

    // Retourne une copie des moyennes.
    public double[] getMean() {
        if (!fitted) {
            throw new IllegalStateException("fit() doit être appelé avant getMean()");
        }
        
        double[] copy = new double[mean.length];
        for (int i = 0; i < mean.length; i++) {
            copy[i] = mean[i];
        }
        
        return copy;
    }

    // Retourne une copie des écarts-types.
    public double[] getStd() {
        if (!fitted) {
            throw new IllegalStateException("fit() doit être appelé avant getStd()");
        }
        double[] copy = new double[std.length];
        for (int i = 0; i < std.length; i++) {
            copy[i] = std[i];
        }
        return copy;
    }
}