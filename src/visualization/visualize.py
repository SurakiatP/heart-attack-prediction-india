import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_names):
    """
    Plot the feature importances for a tree-based model.
    
    Parameters:
        model: Trained model with feature_importances_ attribute.
        feature_names (list): List of feature names.
    """
    importances = model.feature_importances_
    
    # Check if the provided feature_names length matches model's features
    if len(feature_names) != len(importances):
        print(f"Warning: Provided feature_names length ({len(feature_names)}) does not match "
              f"the number of features in the model ({len(importances)}). Using default feature names.")
        feature_names = [f"f{i}" for i in range(len(importances))]
    
    # Sort features by importance
    indices = sorted(range(len(importances)), key=lambda k: importances[k], reverse=True)
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = [importances[i] for i in indices]

    plt.figure(figsize=(10,6))
    plt.title("Feature Importances")
    plt.bar(range(len(sorted_importances)), sorted_importances, align="center")
    plt.xticks(range(len(sorted_importances)), sorted_features, rotation=90)
    plt.tight_layout()
    
    # Create output directory if not exists
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/feature_importance.png"
    
    # Save the plot to file
    plt.savefig(output_path)
    print(f"Feature importance plot saved to {output_path}")
    
    # Optionally, you may call plt.show() if running interactively
    # plt.show()

if __name__ == "__main__":
    import os
    import pickle
    model_path = "models/rf_model.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Example: manually provided feature names (update if available)
    feature_names = ["Age", "Diabetes", "Hypertension", "Gender_Male", "State_Name_Assam", "State_Name_Rajasthan", "Age_Hypertension"]
    plot_feature_importance(model, feature_names)
