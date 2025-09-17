# Copied from maveric/artifact/mro/ue_tracks_generation_helper.py

from typing import Dict, Tuple

from . import constants as C


class UETracksGenerationHelper:
    @staticmethod
    def get_ue_class_distribution_count(ue_tracks_generation_params: Dict) -> Tuple[int, int, int, int]:
        stationary_count = ue_tracks_generation_params[C.UE_CLASS_DISTRIBUTION][C.STATIONARY][C.COUNT]
        pedestrian_count = ue_tracks_generation_params[C.UE_CLASS_DISTRIBUTION][C.PEDESTRIAN][C.COUNT]
        cyclist_count = ue_tracks_generation_params[C.UE_CLASS_DISTRIBUTION][C.CYCLIST][C.COUNT]
        car_count = ue_tracks_generation_params[C.UE_CLASS_DISTRIBUTION][C.CAR][C.COUNT]
        return stationary_count, pedestrian_count, cyclist_count, car_count

    @staticmethod
    def get_ue_class_distribution_velocity(
        ue_tracks_generation_params: Dict, simulation_time_interval: float
    ) -> Tuple[float, float, float, float]:
        stationary_velocity = (
            ue_tracks_generation_params[C.UE_CLASS_DISTRIBUTION][C.STATIONARY][C.VELOCITY]
            * simulation_time_interval
        )
        pedestrian_velocity = (
            ue_tracks_generation_params[C.UE_CLASS_DISTRIBUTION][C.PEDESTRIAN][C.VELOCITY]
            * simulation_time_interval
        )
        cyclist_velocity = (
            ue_tracks_generation_params[C.UE_CLASS_DISTRIBUTION][C.CYCLIST][C.VELOCITY]
            * simulation_time_interval
        )
        car_velocity = (
            ue_tracks_generation_params[C.UE_CLASS_DISTRIBUTION][C.CAR][C.VELOCITY]
            * simulation_time_interval
        )
        return stationary_velocity, pedestrian_velocity, cyclist_velocity, car_velocity

    @staticmethod
    def get_ue_class_distribution_velocity_variances(
        ue_tracks_generation_params: Dict, simulation_time_interval: float
    ) -> Tuple[float, float, float, float]:
        stationary_velocity_variance = (
            ue_tracks_generation_params[C.UE_CLASS_DISTRIBUTION][C.STATIONARY][C.VELOCITY_VARIANCE]
            * simulation_time_interval
        )
        pedestrian_velocity_variance = (
            ue_tracks_generation_params[C.UE_CLASS_DISTRIBUTION][C.PEDESTRIAN][C.VELOCITY_VARIANCE]
            * simulation_time_interval
        )
        cyclist_velocity_variance = (
            ue_tracks_generation_params[C.UE_CLASS_DISTRIBUTION][C.CYCLIST][C.VELOCITY_VARIANCE]
            * simulation_time_interval
        )
        car_velocity_variances = (
            ue_tracks_generation_params[C.UE_CLASS_DISTRIBUTION][C.CAR][C.VELOCITY_VARIANCE]
            * simulation_time_interval
        )

        return (
            stationary_velocity_variance,
            pedestrian_velocity_variance,
            cyclist_velocity_variance,
            car_velocity_variances,
        )

