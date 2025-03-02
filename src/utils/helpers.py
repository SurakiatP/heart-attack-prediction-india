import pickle

def save_pickle(obj, file_path):
    """
    Save an object to a pickle file.
    
    Parameters:
        obj: The object to be saved.
        file_path (str): Path to save the pickle file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(file_path):
    """
    Load an object from a pickle file.
    
    Parameters:
        file_path (str): Path to the pickle file.
    
    Returns:
        The loaded object.
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def print_separator():
    """
    Print a separator line.
    """
    print("-" * 50)
