package ml.preprocessing;

// Normalisation Min-Max pour mettre les données à l'échelle [0, 1].
public class MinMaxScaler implements Preprocessor {
    private double[] min;    // Valeurs minimales.
    private double[] max;    // Valeurs maximales.
    private boolean fitted = false; // État d'entraînement.

    // Calcule les min et max par colonne (ignore la cible y).
    @Override
    public void fit(double[][] dataset) {
        int rows = dataset.length;
        if (rows == 0) return;
        
        int cols = dataset[0].length - 1; 
        min = new double[cols];
        max = new double[cols];

        for (int j = 0; j < cols; j++) {
            min[j] = dataset[0][j];
            max[j] = dataset[0][j];
            for (int i = 1; i < rows; i++) {
                if (dataset[i][j] < min[j]) min[j] = dataset[i][j];
                if (dataset[i][j] > max[j]) max[j] = dataset[i][j];
            }
        }
        fitted = true;
    }

    // Transforme les données selon la formule : (x - min) / (max - min).
    @Override
    public double[][] transform(double[][] dataset) {
        if (!fitted) {
            throw new IllegalStateException("fit() doit être appelé avant transform().");
        }

        int rows = dataset.length;
        int cols = dataset[0].length;
        double[][] newData = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols - 1; j++) {
                if (max[j] == min[j]) {
                    newData[i][j] = 0.0;
                } else {
                    newData[i][j] = (dataset[i][j] - min[j]) / (max[j] - min[j]);
                }
            }
            newData[i][cols - 1] = dataset[i][cols - 1];
        }
        return newData;
    }

    // Entraîne et transforme en une étape.
    @Override
    public double[][] fitTransform(double[][] dataset) {
        fit(dataset);
        return transform(dataset);
    }
}