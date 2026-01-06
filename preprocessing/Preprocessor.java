package ml.preprocessing;

// Interface pour préparer les données.
public interface Preprocessor { 
    
    // Calcule les valeurs (min, max, etc.) pendant l'entrainement.
    void fit(double[][] dataset); 

    // Modifie les données selon les calculs faits au début.
    double[][] transform(double[][] dataset); 

    // Fait les deux étapes d'un coup pour aller plus vite.
    double[][] fitTransform(double[][] dataset);
}