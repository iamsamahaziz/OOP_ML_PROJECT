package ml.core;

// Classe de base abstraite pour tous les modèles de Machine Learning.
public abstract class MLModel {
    protected String name; 

    // Constructeur avec le nom du modèle.
    public MLModel(String name) {
        this.name = name; 
    }

    // Affiche l'état de préparation du modèle.
    public void printStatus() {
        System.out.println("Modèle : " + name + " (prêt)"); 
    }

    // Entraîne le modèle sur un dataset.
    public abstract void train(double[][] dataset); 
    
    // Prédit un résultat pour une seule entrée.
    public abstract double predict(double[] input); 
    
    // Prédit les résultats pour un ensemble d'entrées.
    public double[] predict(double[][] inputs) {
        double[] predictions = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            predictions[i] = predict(inputs[i]);
        }
        return predictions;
    }
    
    // Évalue la performance sur un jeu de test.
    public abstract double score(double[][] testSet);

    // Retourne le nom du modèle.
    public String getName() { 
        return name; 
    }
}