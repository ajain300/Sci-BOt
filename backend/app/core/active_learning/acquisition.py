from typing import List, Tuple, Dict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from .utils.k_medoids import k_medoids
import pandas as pd
import logging


from .dataset import AL_Dataset
from .models import GaussianProcessModel, BaseModel
from .targets import Target

# TODO MUST ADD qNEHVI implementation, check botorch compatitbility

logger = logging.getLogger(__name__)

class AcquisitionFunction:
    def __init__(self, model: BaseModel, dataset: AL_Dataset, targets: Dict[str, Target] = None, **kwargs):
        self.model = model
        self.dataset = dataset
        self.targets = targets if targets else dataset.targets
        self.kwargs = kwargs
        
    
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
                logger.info(f"Predicting outputs for property: {target_obj.name}")
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
        logger.info("Initializing Diversity_Batch acquisition function.")

    def recommend(self, N_samples = 10, beta = 0.25, property = None, **kwargs):
        """
        Implements DMB AL acquisition from https://arxiv.org/pdf/1901.05954
        NOTE if using a single property it must be specified
        """
        cluster_method = kwargs.get('cluster_method', 'kmedoids')
        mean, std_dict, candidates, og_candidate_col = self._get_predictions(property = property)
        std = [std_dict[prop] for prop in std_dict]
        std = np.sum(std, axis = 0)
        
        # Top beta % samples with uncertainty
        k = round(std.shape[0]*beta)
        top_ind = np.argsort(std)[-k:]
        top_uncertain_samples = candidates.iloc[top_ind].to_numpy()

        if cluster_method == 'kmeans':
            # Kmeans clustering for diversity samples
            kmeans = KMeans(n_clusters=N_samples)
            kmeans.fit(top_uncertain_samples)
        
            centers = kmeans.cluster_centers_

            closest_index, _ = pairwise_distances_argmin_min(centers, top_uncertain_samples)
            chosen_ratios = top_uncertain_samples[closest_index]
        elif cluster_method == 'kmedoids':
            # Kmedoids clustering for diversity samples
            closest_index, _ = k_medoids(top_uncertain_samples, N_samples)
            chosen_ratios = top_uncertain_samples[closest_index]

        rec_samples = pd.DataFrame(chosen_ratios, columns=og_candidate_col)

        # Unnormalize the predictions
        for prop, preds in mean.items():
            mean[prop], std_dict[prop] = self.dataset.unnormalize(preds, self.targets[prop].X_columns[0], data_std = std_dict[prop] ** 0.5)
        
        # Get acquired prediction means and std
        acq_sample_mean = {}
        acq_sample_std = {}
        for prop, preds in mean.items():
            target = self.targets[prop]
            mean[prop], std_dict[prop] = self.dataset.unnormalize(
                preds, 
                target.X_columns[0],
                data_std=std_dict[prop] ** 0.5
            )
            
        self.acq_stats = {
            'avg_mean': {prop: np.mean(preds) for prop, preds in mean.items()},
            'max_mean': {prop: np.max(preds) for prop, preds in mean.items()},
            'min_mean': {prop: np.min(preds) for prop, preds in mean.items()},
            'avg_std': {prop: np.mean(std_dict[prop]) for prop in std_dict},
            'max_std': {prop: np.max(std_dict[prop]) for prop in std_dict},
            'min_std': {prop: np.min(std_dict[prop]) for prop in std_dict}
        }

        return rec_samples, acq_sample_mean, acq_sample_std
    
class NaiveMOFunction(AcquisitionFunction):
    def __init__(self, model: BaseModel, dataset: AL_Dataset, targets: Dict[str, Target], **kwargs):
        super().__init__(model, dataset, **kwargs)
        self.targets = targets
        logger.info("Initializing NaiveMOFunction acquisition function.")

    def get_score(self, predictions_dict : Dict[str, np.ndarray]) -> float:
        total_score = 0
        for key, target_object in self.targets.items():
            # Assuming each target has a method to calculate its score
            score = target_object.score(predictions_dict[target_object.name])  # You may need to implement this method in the Target class
            total_score += score
        return total_score
    
    def recommend(self, N_samples = 10, property = None, **kwargs):
        # Get the predictions
        mean, std, candidates, candidate_col = self._get_predictions()
        
        # Unnormalize the predictions
        for prop, preds in mean.items():
            mean[prop], std[prop] = self.dataset.unnormalize(preds, self.targets[prop].X_columns[0], data_std = std[prop])
        
        logger.info(f"mean_pred: {mean}")
        score = np.zeros_like(list(mean.values())[0])
        for target_name, target in self.targets.items():

            # print(target_name)
            # print("mean", mean[target_name])
            # print("max", max(mean[target_name]))
            # print("mean", max(mean[target_name]))
            # print(target.score(mean[target_name], std[target_name]))
            # print(max(target.score(mean[target_name], std[target_name])))
            score += target.score(mean[target_name], std[target_name])

        logger.info(f"score {score}")

        top_ind = np.argsort(score)[-N_samples:]
        # print(candidates)
        # print(top_ind)
        
        # Get acquired samples and predictions
        acquired_samples = candidates.iloc[top_ind].to_numpy()
        acq_sample_mean = {}
        acq_sample_std = {}
        for prop, preds in mean.items():
                acq_sample_mean[prop], acq_sample_std[prop] = mean[prop][top_ind], std[prop][top_ind]
        
        # print(acq_sample_mean, acq_sample_std, score[top_ind])
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
        logger.info("Initializing NaiveMOFunction acquisition function.")

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
        

        
