import yaml
import numpy as np
import logging
import os,sys
import pickle

logger = logging.getLogger(__name__)



def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise Exception(e) 
    
def eqLinear(OddsAtAnchor,Anchor=600,PDO=20):
    alpha=Anchor- PDO/np.log(2)* np.log(OddsAtAnchor)
    beta= PDO/np.log(2)
    return {'alpha':np.round(alpha,4),'beta':np.round(beta,4)}


def three_digit_score(prob_series, alpha, beta):
    """
    Efficiently convert probability column to three-digit scores.
    
    Parameters:
    -----------
    prob_series : pd.Series or np.ndarray
        Column of probabilities to convert and this probability has to be default probabilities
    alpha : float
        Intercept/offset term
    beta : float
        Scaling coefficient
    
    Returns:
    --------
    np.ndarray
        Array of rounded three-digit scores
    """
    # Vectorized log-odds transformation
    log_odds = np.log((1 - prob_series) / prob_series)
    
    # Vectorized score calculation
    scores = (log_odds * beta) + alpha
    
    # Vectorized rounding
    return np.round(scores).astype(int)

    
def save_object(file_path: str, obj: object) -> None:
    try:
        logger.info("Entered the save_object method of Utils class")
        
        # Get the directory part of the file path
        directory = os.path.dirname(file_path)
        
        # Create the folder if it exists and is not empty
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Save the object using pickle
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.info(f"Successfully saved object to {file_path}")

    except Exception as e:
        raise Exception(f"An error occurred while saving the object: {e}")
    

def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
            logger.info(f"Successfully loaded object from {file_path}")
            return obj
    except Exception as e:
        raise Exception(e)
    
