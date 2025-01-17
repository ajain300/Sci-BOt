from typing import List, Tuple, Dict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import pandas as pd
import logging


from .dataset import AL_Dataset
from .models import GaussianProcessModel, BaseModel
from .targets import Target

logger = logging.getLogger(__name__)

class AcquisitionFunction: 
    def __init__(self, model : BaseModel, dataset : AL_Dataset, **kwargs):
        self.model = model
        self.dataset = dataset
        self.kwargs = kwargs
        
        self.logger = kwargs.get('logger', logging.getLogger(__name__))
        
    
    def recommend(self, X, model, **kwargs):
        raise NotImplementedError("Method 'train' must be implemented in subclasses")

    def _get_predictions(self, property = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
        # Generate candidate space
        candidates = self.dataset.generate_candidate_space(n_compositions = self.kwargs.get('__gen__comp_n', 11))
        og_candidate_col = candidates.columns
        
        # If property is given then do this 
        mean = {}
        std = {}
        if property is None:
            for target_name, target_obj in self.dataset.targets.items():
                self.logger.info(f"Predicting outputs for property: {target_obj.name}")
                m, s = self.model.inference(candidates.copy(), property = target_obj.name)
                mean[target_obj.name] = m
                std[target_obj.name] = s
        else:
            mean, std = self.model.inference(candidates.copy(), property = property)
            mean = {property: mean}
            std = {property: std}
            
        return mean, std, candidates, og_candidate_col
        
class Diversity_Batch(AcquisitionFunction):
    def __init__(self, model : BaseModel, dataset : AL_Dataset, **kwargs):
        super().__init__(model, dataset, **kwargs)
        self.logger.info("Initializing Diversity_Batch acquisition function.")

    
    def recommend(self, N_samples = 10, beta = 0.50, property = None, **kwargs):
        """
        Implements DMB AL acquisition from https://arxiv.org/pdf/1901.05954
        NOTE if using a single property it must be specified
        """
        # Generate candidate space
        # candidates = self.dataset.generate_candidate_space()
        # og_candidate_col = candidates.columns
        
        # # If property is given then do this 
        # std_list = []
        # if property is None:
        #     for target in self.dataset.targets:
        #         logger.info(f"Predicting outputs for property: {target}")
        #         mean, std = self.model.inference(candidates.copy(), property = target)
        #         std_list.append(std)
        #     std = np.sum(std_list, axis = 0)
        # else:
        #     print(property)
        #     mean, std = self.model.inference(candidates.copy(), property = property)
        
        mean, std, candidates, og_candidate_col = self._get_predictions(property = property)
        std = [std[prop] for prop in std]
        std = np.sum(std, axis = 0)
        
        # Top beta % samples with uncertainty
        k = round(std.shape[0]*beta)
        top_ind = np.argsort(std)[-k:]
        top_uncertain_samples = candidates.iloc[top_ind].to_numpy()

        # Kmeans clustering for diversity samples
        kmeans = KMeans(n_clusters=N_samples)
        kmeans.fit(top_uncertain_samples)
        
        centers = kmeans.cluster_centers_

        closest_index, _ = pairwise_distances_argmin_min(centers, top_uncertain_samples)
        closest_points = top_uncertain_samples[closest_index]
        rec_samples = pd.DataFrame(closest_points, columns=og_candidate_col)
        return rec_samples
    
class NaiveMOFunction(AcquisitionFunction):
    def __init__(self, model: BaseModel, dataset: AL_Dataset, targets: Dict[str, Target], **kwargs):
        super().__init__(model, dataset, **kwargs)
        self.targets = targets
        self.logger.info("Initializing NaiveMOFunction acquisition function.")

    def get_score(self, predictions_dict : Dict[str, np.ndarray]) -> float:
        total_score = 0
        for key, target_object in self.targets.items():
            # Assuming each target has a method to calculate its score
            score = target_object.score(predictions_dict[target_object.name])  # You may need to implement this method in the Target class
            total_score += score
        return total_score
    
    def recommend(self, N_samples = 10, property = None, **kwargs):
        # Get the predictions
        print("naive mo scoring")
        mean, std, candidates, candidate_col = self._get_predictions()
        
        # Unnormalize the predictions
        for prop, preds in mean.items():
            mean[prop], std[prop] = self.dataset.unnormalize(preds, self.targets[prop].X_columns[0], data_std = std[prop])
        
        print("mean_pred", mean)
        score = np.zeros_like(list(mean.values())[0])
        for target_name, target in self.targets.items():

            print(target_name)
            print("mean", mean[target_name])
            print("max", max(mean[target_name]))
            print("mean", max(mean[target_name]))
            print(target.score(mean[target_name], std[target_name]))
            print(max(target.score(mean[target_name], std[target_name])))
            score += target.score(mean[target_name], std[target_name])

        print("score", score)

        top_ind = np.argsort(score)[-N_samples:]
        print(candidates)
        print(top_ind)
        
        # Get acquired samples and predictions
        acquired_samples = candidates.iloc[top_ind].to_numpy()
        acq_sample_mean = {}
        acq_sample_std = {}
        for prop, preds in mean.items():
                acq_sample_mean[prop], acq_sample_std[prop] = mean[prop][top_ind], std[prop][top_ind]
        
        print(acq_sample_mean, acq_sample_std, score[top_ind])
        acquired_samples = pd.DataFrame(acquired_samples, columns=candidate_col)
        
        return acquired_samples, acq_sample_mean, acq_sample_std


class Score_Diversity_Batch(AcquisitionFunction):
    """
    Samples that fit the score criteria but use the clustering from the diversity.

    Args:
        AcquisitionFunction (_type_): _description_
    """
    def __init__(self, model: BaseModel, dataset: AL_Dataset, targets: Dict[str, Target], **kwargs):
        super().__init__(model, dataset, **kwargs)
        self.targets = targets
        self.logger.info("Initializing NaiveMOFunction acquisition function.")

    def get_score(self, predictions_dict : Dict[str, np.ndarray]) -> float:
        total_score = 0
        for key, target_object in self.targets.items():
            # Assuming each target has a method to calculate its score
            score = target_object.score(predictions_dict[target_object.name])  # You may need to implement this method in the Target class
            total_score += score
        return total_score
    
    def recommend(self, N_samples = 10, property = None, **kwargs):
        # Get the predictions
        print("naive mo scoring")
        mean, std, candidates, candidate_col = self._get_predictions()
        for prop_name, prop_obj in self.targets.items():
            print(prop_name)
            print(prop_obj.name)
            print(prop_obj.X_columns)
        
        # Unnormalize the predictions
        for prop, preds in mean.items():
            print(prop)
            mean[prop], std[prop] = self.dataset.unnormalize(preds, self.targets[prop].X_columns[0], data_std = std[prop])
        
        print(mean)
        score = np.zeros_like(list(mean.values())[0])
        for target_name, target in self.targets.items():
            print(target.score(mean[target_name], std[target_name]))
            score += target.score(mean[target_name], std[target_name])

        print(score)

        top_ind = np.argsort(score)[-N_samples:]
        print(candidates)
        print(top_ind)

        acquired_samples = candidates.iloc[top_ind].to_numpy()

        print(acquired_samples)
        acquired_samples = pd.DataFrame(acquired_samples, columns=candidate_col)

        return acquired_samples
        

        
