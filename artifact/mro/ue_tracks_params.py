# Copied from maveric/artifact/mro/ue_tracks_params.py

from typing import Dict

from . import constants as C
from .ue_tracks import MobilityClass
from .ue_tracks_generation_helper import UETracksGenerationHelper


class UETracksGenerationParams:
    def __init__(self, params: Dict):
        self.params = params[C.UE_TRACKS_GENERATION][C.PARAMS]
        self.rng_seed = self.params[C.GAUSS_MARKOV_PARAMS][C.RNG_SEED]
        self.num_batches = self.params[C.NUM_BATCHES]
        self.lon_x_dims = self.params[C.GAUSS_MARKOV_PARAMS][C.LON_X_DIMS]
        self.lon_y_dims = self.params[C.GAUSS_MARKOV_PARAMS][C.LON_Y_DIMS]
        self.num_ticks = self.params[C.NUM_TICKS]
        self.num_UEs = self.extract_ue_class_distribution()
        self.alpha = self.params[C.GAUSS_MARKOV_PARAMS][C.ALPHA]
        self.variance = self.params[C.GAUSS_MARKOV_PARAMS][C.VARIANCE]
        self.min_lat = self.params[C.LON_LAT_BOUNDARIES][C.MIN_LAT]
        self.max_lat = self.params[C.LON_LAT_BOUNDARIES][C.MAX_LAT]
        self.min_lon = self.params[C.LON_LAT_BOUNDARIES][C.MIN_LON]
        self.max_lon = self.params[C.LON_LAT_BOUNDARIES][C.MAX_LON]
        self.extract_ue_class_distribution()

    def extract_ue_class_distribution(self) -> int:
        simulation_time_interval = self.params[C.SIMULATION_TIME_INTERVAL]

        (
            stationary_count,
            pedestrian_count,
            cyclist_count,
            car_count,
        ) = UETracksGenerationHelper.get_ue_class_distribution_count(self.params)

        self.num_UEs = stationary_count + pedestrian_count + cyclist_count + car_count

        stationary_distribution = stationary_count / self.num_UEs if self.num_UEs else 0.0
        pedestrian_distribution = pedestrian_count / self.num_UEs if self.num_UEs else 0.0
        cyclist_distribution = cyclist_count / self.num_UEs if self.num_UEs else 0.0
        car_distribution = car_count / self.num_UEs if self.num_UEs else 0.0

        self.mobility_class_distribution = {
            MobilityClass.stationary: stationary_distribution,
            MobilityClass.pedestrian: pedestrian_distribution,
            MobilityClass.cyclist: cyclist_distribution,
            MobilityClass.car: car_distribution,
        }

        (
            stationary_velocity,
            pedestrian_velocity,
            cyclist_velocity,
            car_velocity,
        ) = UETracksGenerationHelper.get_ue_class_distribution_velocity(self.params, simulation_time_interval)

        self.mobility_class_velocities = {
            MobilityClass.stationary: stationary_velocity,
            MobilityClass.pedestrian: pedestrian_velocity,
            MobilityClass.cyclist: cyclist_velocity,
            MobilityClass.car: car_velocity,
        }

        (
            stationary_velocity_variance,
            pedestrian_velocity_variance,
            cyclist_velocity_variance,
            car_velocity_variances,
        ) = UETracksGenerationHelper.get_ue_class_distribution_velocity_variances(
            self.params, simulation_time_interval
        )

        self.mobility_class_velocity_variances = {
            MobilityClass.stationary: stationary_velocity_variance,
            MobilityClass.pedestrian: pedestrian_velocity_variance,
            MobilityClass.cyclist: cyclist_velocity_variance,
            MobilityClass.car: car_velocity_variances,
        }

        return self.num_UEs

