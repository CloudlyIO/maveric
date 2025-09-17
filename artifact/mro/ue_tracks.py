# Copied from maveric/artifact/mro/ue_tracks.py

from enum import Enum
from typing import Dict, Generator, List, Any, Optional
import itertools

import numpy as np
import pandas as pd

from .mobility import gauss_markov
from . import constants as C


class MobilityClass(Enum):
    stationary = "stationary"
    pedestrian = "pedestrian"
    cyclist = "cyclist"
    car = "car"


class UETracksGenerator:
    def __init__(
        self,
        rng: np.random.Generator,
        mobility_class_distribution: Dict[MobilityClass, float],
        mobility_class_velocities: Dict[MobilityClass, float],
        mobility_class_velocity_variances: Dict[MobilityClass, float],
        lon_x_dims: int = 100,
        lon_y_dims: int = 100,
        num_ticks: int = 2,
        num_UEs: int = 2,
        alpha: float = 0.5,
        variance: float = 0.8,
        min_lat: float = -90,
        max_lat: float = 90,
        min_lon: float = -180,
        max_lon: float = 180,
        anchor_loc: Optional[np.ndarray] = None,
        cov_around_anchor: Optional[np.ndarray] = None,
    ):
        self.rng = rng
        self.lon_x_dims = lon_x_dims
        self.lon_y_dims = lon_y_dims
        self.num_ticks = num_ticks
        self.num_UEs = num_UEs
        self.alpha = alpha
        self.variance = variance
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.anchor_loc = anchor_loc
        self.cov_around_anchor = cov_around_anchor
        self.mobility_class_distribution = mobility_class_distribution
        self.mobility_class_velocities = mobility_class_velocities
        self.mobility_class_velocity_variances = mobility_class_velocity_variances

        self.sampled_users_per_mobility_class = self.rng.choice(
            [mc.value for mc in list(self.mobility_class_distribution.keys())],
            size=(self.num_UEs),
            replace=True,
            p=list(self.mobility_class_distribution.values()),
        )

        self.num_users_per_mobility_class: Dict[MobilityClass, int] = {}
        self.velocity_range: Dict[MobilityClass, List[float]] = {}
        self.gauss_markov_models: Dict[MobilityClass, Generator] = {}

        for k in self.mobility_class_distribution.keys():
            self.num_users_per_mobility_class[k] = int(np.count_nonzero(self.sampled_users_per_mobility_class == k.value))
            low = self.mobility_class_velocities[k] - self.mobility_class_velocity_variances[k]
            high = self.mobility_class_velocities[k] + self.mobility_class_velocity_variances[k]
            self.velocity_range[k] = [low, high]

        for k in self.mobility_class_distribution.keys():
            self.gauss_markov_models[k] = gauss_markov(
                rng=self.rng,
                num_users=self.num_users_per_mobility_class[k],
                dimensions=(self.lon_x_dims, self.lon_y_dims),
                velocity_mean=self.rng.uniform(
                    low=self.velocity_range[k][0],
                    high=self.velocity_range[k][1],
                    size=self.num_users_per_mobility_class[k],
                ),
                alpha=self.alpha,
                variance=self.variance,
                anchor_loc=self.anchor_loc,
                cov_around_anchor=self.cov_around_anchor,
            )

    def generate(self) -> Generator:
        while True:
            tracks = []
            for _ in range(0, self.num_ticks):
                xy_lonlat = []
                for k in self.gauss_markov_models.keys():
                    xy = next(self.gauss_markov_models[k])
                    xy_lonlat.extend(xy)
                tracks.append(xy_lonlat)
            yield tracks

    def close(self):
        for k in self.gauss_markov_models.keys():
            self.gauss_markov_models[k].close()

    @staticmethod
    def generate_as_lon_lat_points(
        rng_seed: int,
        lon_x_dims: int,
        lon_y_dims: int,
        num_ticks: int,
        num_batches: int,
        num_UEs: int,
        alpha: float,
        variance: float,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        mobility_class_distribution: Dict[MobilityClass, float],
        mobility_class_velocities: Dict[MobilityClass, float],
        mobility_class_velocity_variances: Dict[MobilityClass, float],
    ) -> Generator[pd.DataFrame, None, None]:
        ue_tracks_generator = UETracksGenerator(
            rng=np.random.default_rng(rng_seed),
            lon_x_dims=lon_x_dims,
            lon_y_dims=lon_y_dims,
            num_ticks=num_ticks,
            num_UEs=num_UEs,
            alpha=alpha,
            variance=variance,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            mobility_class_distribution=mobility_class_distribution,
            mobility_class_velocities=mobility_class_velocities,
            mobility_class_velocity_variances=mobility_class_velocity_variances,
        )

        for _num_batches, xy_batches in enumerate(ue_tracks_generator.generate()):
            ue_tracks_dataframe_dict: Dict[Any, Any] = {}

            mock_ue_id: List[int] = []
            ticks: List[int] = []
            lon: List[float] = []
            lat: List[float] = []

            tick = 0
            for xy_batch in xy_batches:
                lon_lat_pairs = []
                for point in xy_batch:
                    lon_val = min_lon + ((max_lon - min_lon) / lon_x_dims) * point[0]
                    lat_val = min_lat + ((max_lat - min_lat) / lon_y_dims) * point[1]
                    lon_lat_pairs.append((lon_val, lat_val))

                lon.extend(p[0] for p in lon_lat_pairs)
                lat.extend(p[1] for p in lon_lat_pairs)
                mock_ue_id.extend([i for i in range(num_UEs)])
                ticks.extend(list(itertools.repeat(tick, num_UEs)))
                tick += 1

            ue_tracks_dataframe_dict[C.MOCK_UE_ID] = mock_ue_id
            ue_tracks_dataframe_dict[C.LONGITUDE] = lon
            ue_tracks_dataframe_dict[C.LATITUDE] = lat
            ue_tracks_dataframe_dict[C.TICK] = ticks

            yield pd.DataFrame(ue_tracks_dataframe_dict)

            num_batches -= 1
            if num_batches == 0:
                break

