import os
import random
import math
from typing import Dict, List, Union, Optional
import numpy as np
import pandas as pd
import itertools
from copy import deepcopy
from .models import Pipeline


class Optimizer:
    def __init__(self,
        task_data: Pipeline,
        complete_profile: bool,
        only_measured_profiles: bool,
    )-> None:
        self.complete_profile = complete_profile
        self.only_measured_profiles = only_measured_profiles
        self.task_data = task_data
    
    def sla_is_met(self) -> bool:
        return self.task_data.pipeline_latency < self.task_data.sla

    def can_sustain_load(self, arrival_rate: int) -> bool:
        """
        whether the existing config can sustain a load
        """
        for task in self.task_data.inference_graph:
            if arrival_rate > task.throughput_all_replicas:
                return False
        return True
    
    def constraints(self, arrival_rate: int) -> bool:
        """
        whether the constraints are met or not
        """
        if self.sla_is_met() and self.can_sustain_load(arrival_rate=arrival_rate):
            return True
        return False
    
    def batch_size_objective(self) -> float:
        """
        batch objecive of the pipeline
        """
        max_batch = 0
        for task in self.pipeline.inference_graph:
            max_batch += task.batch
        return max_batch

    def objective(self, alpha: float, beta: float) -> Dict[str, float]:
        """
        objective function of the pipeline
        """
        objectives = {}
        objectives["accuracy_objective"] = alpha * self.accuracy_objective()
        objectives["batch_objective"] = beta * self.batch_objective()
        objectives["objective"] = (
            objectives["accuracy_objective"]
            - objectives["batch_objective"]
        )
        return objectives
    
    def brute_force(
        self,
        scaling_cap: int,
        alpha: float,
        beta: float,
        gamma: float,
        arrival_rate: int,
        num_state_limit: int = None,
    ) -> pd.DataFrame:
        states = self.all_states(
            check_constraints=True,
            scaling_cap=scaling_cap,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            arrival_rate=arrival_rate,
            num_state_limit=num_state_limit,
        )
        optimal = states[states["objective"] == states["objective"].max()]
        return optimal

    def all_states(
        self,
        scaling_cap: int,
        alpha: float,
        beta: float,
        gamma: float,
        check_constraints: bool,
        arrival_rate: int,
        num_state_limit: int = None,
    ) -> pd.DataFrame:
        

    

