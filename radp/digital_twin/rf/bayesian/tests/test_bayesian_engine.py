# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This test demonstrates how the BayesianDigitalTwin (a RADP RF Digital Twin model)
is trained and inferred on, using small sample data.
"""

import logging
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from radp.utility.simulation_utils import seed_everything
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin
from radp.digital_twin.utils import constants
from radp.digital_twin.utils.gis_tools import GISTools


# Hard-coded sample site config data and UE data
def get_sample_site_config_and_ue_data():
    site_configs_df = pd.DataFrame(
        {
            "cell_id": ["Cell1", "Cell2", "Cell3"],
            "cell_az_deg": [0, 120, 240],
            "cell_el_deg": [0, 0, 0],
            "cell_lat": [35.690555, 35.690555, 35.690555],
            "cell_lon": [139.69194, 139.69194, 139.69194],
            "cell_carrier_freq_mhz": [2100, 2100, 2100],
        }
    )

    # synthetic sample UE data : 5 data points

    ue_data_df = pd.DataFrame(
        {
            "cell_id": [
                "Cell1",
                "Cell1",
                "Cell1",
                "Cell1",
                "Cell1",
                "Cell2",
                "Cell2",
                "Cell2",
                "Cell2",
                "Cell2",
                "Cell3",
                "Cell3",
                "Cell3",
                "Cell3",
                "Cell3",
            ],
            "avg_rsrp": [
                -80,
                -70,
                -75,
                -72,
                -71,
                -100,
                -101,
                -102,
                -99,
                -98,
                -77,
                -78,
                -79,
                -80,
                -79,
            ],
            "lon": [
                139.699058,
                139.707889,
                139.700023,
                139.702645,
                139.702645,
                139.707067,
                139.700519,
                139.701644,
                139.701644,
                139.701644,
                139.702793,
                139.703664,
                139.704312,
                139.704312,
                139.70469,
            ],
            "lat": [
                35.644327,
                35.64781,
                35.643857,
                35.645913,
                35.64591,
                35.647007,
                35.644816,
                35.645196,
                35.645196,
                35.645198,
                35.645571,
                35.645876,
                35.646208,
                35.646209,
                35.645790,
            ],
        }
    )
    return site_configs_df, ue_data_df


def get_x_max_and_x_min():
    """
    This provides reasonable ranges for typical parameters used in RF Digital Twin.
    This is useful for normalizing values before training.
    """

    x_max = {
        "cell_el_deg": 50,
        "cell_lat": 90,
        "cell_lon": 180,
        "distance": 5,  # log(1 + max distance of 50000 meters)
        "relative_bearing": 360,
        "relative_tilt": 180,
        "relative_tilt_squared": 32400,
    }
    x_min = {
        "cell_el_deg": -10,
        "cell_lat": -90,
        "cell_lon": -180,
        "distance": 0,
        "relative_bearing": 0,
        "relative_tilt": -90,
        "relative_tilt_squared": 0,
    }

    return x_max, x_min


def augment_ue_data(ue_data_df: pd.DataFrame, site_configs_df: pd.DataFrame):
    """
    This method takes in UE data and site config and adds to the UE data
    as well as engineers new training features in (e.g. log distance and
    relative bearing)
    """

    for i in ue_data_df.index:
        site_config = site_configs_df[
            site_configs_df.cell_id == ue_data_df.at[i, "cell_id"]
        ]
        ue_data_df.at[i, "cell_id"] = site_config.cell_id.values[0]
        ue_data_df.at[i, "loc_x"] = ue_data_df.at[i, "lon"]
        ue_data_df.at[i, "loc_y"] = ue_data_df.at[i, "lat"]
        ue_data_df.at[i, "cell_lat"] = site_config.cell_lat.values[0]
        ue_data_df.at[i, "cell_lon"] = site_config.cell_lon.values[0]
        ue_data_df.at[i, "cell_az_deg"] = site_config.cell_az_deg.values[0]
        ue_data_df.at[i, "cell_el_deg"] = site_config.cell_el_deg.values[0]
        ue_data_df.at[
            i, "cell_carrier_freq_mhz"
        ] = site_config.cell_carrier_freq_mhz.values[0]

        ue_data_df.at[i, "log_distance"] = GISTools.get_log_distance(
            ue_data_df.at[i, "cell_lat"],
            ue_data_df.at[i, "cell_lon"],
            ue_data_df.at[i, "lat"],
            ue_data_df.at[i, "lon"],
        )
        ue_data_df.at[i, "relative_bearing"] = GISTools.get_relative_bearing(
            ue_data_df.at[i, "cell_az_deg"],
            ue_data_df.at[i, "cell_lat"],
            ue_data_df.at[i, "cell_lon"],
            ue_data_df.at[i, "lat"],
            ue_data_df.at[i, "lon"],
        )


def split_training_and_test_data(ue_data_df: pd.DataFrame, test_size: float):
    """
    This function splits the UE data into training and test data according to the defined test size
    It returns the training and test data mapped to cell_id
    """
    cell_id_ue_data_map = {k: v for k, v in ue_data_df.groupby("cell_id")}

    cell_id_training_data_map = {}
    cell_id_test_data_map = {}

    all_coords = pd.DataFrame(columns=["lat", "lon"])

    for cell_id, ue_data in cell_id_ue_data_map.items():
        train, test = train_test_split(ue_data, test_size=test_size)
        cell_id_training_data_map[cell_id] = train
        cell_id_test_data_map[cell_id] = test
        coords = test[["lat", "lon"]]
        all_coords = pd.merge(all_coords, coords, how="outer", on=["lat", "lon"])

    return cell_id_training_data_map, cell_id_test_data_map


class TestBayesianDigitalTwin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything(1)

        # get sample data
        site_configs_df, ue_data_df = get_sample_site_config_and_ue_data()

        site_configs_df.reset_index(drop=True, inplace=True)

        # feature engineering -- add relative bearing and distance
        augment_ue_data(ue_data_df, site_configs_df)

        logging.info(ue_data_df)

        # split into training/test
        (
            cls.cell_id_training_data_map,
            cls.cell_id_test_data_map,
        ) = split_training_and_test_data(ue_data_df, 0.2)

        # Generate test data expected_loss_vs_iter, expected_mae and expected_mape with seed_everything(1)
        cls.expected_loss_vs_iter = {}
        cls.expected_mae = {}
        cls.expected_mape = {}

    def produce_results_rf_bayesian_digital_twin(self):
        # train
        bayesian_digital_twin_map = {}

        x_max, x_min = get_x_max_and_x_min()

        for cell_id, training_data in self.cell_id_training_data_map.items():
            bayesian_digital_twin_map[cell_id] = BayesianDigitalTwin(
                data_in=[training_data],
                x_columns=[
                    constants.CELL_EL_DEG,
                    constants.LOG_DISTANCE,
                    constants.RELATIVE_BEARING,
                ],
                y_columns=["avg_rsrp"],
                x_max=x_max,
                x_min=x_min,
            )
            loss_vs_iter = bayesian_digital_twin_map[cell_id].train_distributed_gpmodel(
                maxiter=20,
                stopping_threshold=1e-5,
            )
            logging.info(
                f"\nTrained {len(loss_vs_iter)} epochs of Bayesian Digital Twin (Gaussian Process Regression) "
                f"on {len(training_data)} data points"
                f" for cell {cell_id}, with min learning loss {min(loss_vs_iter):0.5f}, "
                f"avg learning loss {np.mean(loss_vs_iter):0.5f} and final learning loss {loss_vs_iter[-1]:0.5f}"
            )
            self.expected_loss_vs_iter[cell_id] = loss_vs_iter.tolist()

        # predict/test
        for cell_id, testing_data in self.cell_id_test_data_map.items():
            (pred_means, _) = bayesian_digital_twin_map[
                cell_id
            ].predict_distributed_gpmodel(prediction_dfs=[testing_data])
            MAE = abs(testing_data.avg_rsrp - pred_means[0]).mean()
            # mean absolute percentage error
            MAPE = (
                100
                * abs(
                    (testing_data.avg_rsrp - pred_means[0]) / testing_data.avg_rsrp
                ).mean()
            )
            logging.info(
                f"cell_id = {cell_id}, MAE = {MAE:0.5f} dB, MAPE = {MAPE:0.5f} %,"
                "# test points = {len(testing_data.avg_rsrp)}"
            )
            self.expected_mae[cell_id] = MAE
            self.expected_mape[cell_id] = MAPE

    def test_reproducibility_of_results_for_bayesian_digital_twin(self):
        # produce results and re_run to verify reproducibility of result using seed(1)
        self.produce_results_rf_bayesian_digital_twin()
        # train
        bayesian_digital_twin_map = {}

        x_max, x_min = get_x_max_and_x_min()

        for cell_id, training_data in self.cell_id_training_data_map.items():
            bayesian_digital_twin_map[cell_id] = BayesianDigitalTwin(
                data_in=[training_data],
                x_columns=[
                    constants.CELL_EL_DEG,
                    constants.LOG_DISTANCE,
                    constants.RELATIVE_BEARING,
                ],
                y_columns=["avg_rsrp"],
                x_max=x_max,
                x_min=x_min,
            )
            loss_vs_iter = bayesian_digital_twin_map[cell_id].train_distributed_gpmodel(
                maxiter=20,
                stopping_threshold=1e-5,
            )
            logging.info(
                f"\nTrained {len(loss_vs_iter)} epochs of Bayesian Digital Twin (Gaussian Process Regression) "
                f"on {len(training_data)} data points"
                f" for cell {cell_id}, with min learning loss {min(loss_vs_iter):0.5f}, "
                f"avg learning loss {np.mean(loss_vs_iter):0.5f} and final learning loss {loss_vs_iter[-1]:0.5f}"
            )
            self.assertEqual(self.expected_loss_vs_iter[cell_id], loss_vs_iter.tolist())

        # predict/test
        for cell_id, testing_data in self.cell_id_test_data_map.items():
            (pred_means, _) = bayesian_digital_twin_map[
                cell_id
            ].predict_distributed_gpmodel(prediction_dfs=[testing_data])
            MAE = abs(testing_data.avg_rsrp - pred_means[0]).mean()
            # mean absolute percentage error
            MAPE = (
                100
                * abs(
                    (testing_data.avg_rsrp - pred_means[0]) / testing_data.avg_rsrp
                ).mean()
            )
            logging.info(
                f"cell_id = {cell_id}, MAE = {MAE:0.5f} dB, MAPE = {MAPE:0.5f} %,"
                "# test points = {len(testing_data.avg_rsrp)}"
            )
            self.assertEqual(self.expected_mae[cell_id], MAE)
            self.assertEqual(self.expected_mape[cell_id], MAPE)


def _bearing_gap_deg(a, b):
    """Absolute angular difference in degrees, wrapped to [0, 180]."""
    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)


class TestBearingConvention(unittest.TestCase):
    """Training, prediction-service and prediction-frame features must use the same bearing."""

    def setUp(self):
        self.site_configs_df, self.ue_data_df = get_sample_site_config_and_ue_data()
        self.site_configs_df["hTx"] = 30.0
        self.site_configs_df["hRx"] = 1.5
        template = self.ue_data_df[["lon", "lat"]].rename(columns={"lon": "loc_x", "lat": "loc_y"})
        self.prediction_frames = BayesianDigitalTwin.create_prediction_frames(self.site_configs_df, template)

    def _assert_matches_prediction_frames(self, preprocessed):
        for cell_id, frame in preprocessed.items():
            expected = self.prediction_frames[cell_id].loc[frame.index, constants.RELATIVE_BEARING]
            gap = _bearing_gap_deg(frame[constants.RELATIVE_BEARING], expected)
            self.assertLess(gap.max(), 1e-6, f"{cell_id}: bearing off by up to {gap.max():.3f} deg")

    def test_training_preprocess_matches_prediction_frames(self):
        preprocessed = BayesianDigitalTwin.preprocess_ue_training_data(
            self.ue_data_df.copy(), self.site_configs_df.copy()
        )
        self._assert_matches_prediction_frames(preprocessed)

    def test_prediction_preprocess_matches_prediction_frames(self):
        preprocessed = BayesianDigitalTwin.preprocess_ue_prediction_data(
            self.ue_data_df.copy(),
            self.site_configs_df[["cell_id", "cell_el_deg"]].copy(),
            self.site_configs_df.copy(),
        )
        self._assert_matches_prediction_frames(preprocessed)


class TestKernel(unittest.TestCase):
    def _twin(self, **kwargs):
        site_configs_df, ue_data_df = get_sample_site_config_and_ue_data()
        augment_ue_data(ue_data_df, site_configs_df)
        cell_data = ue_data_df[ue_data_df.cell_id == "Cell1"]
        x_columns = [constants.CELL_EL_DEG, constants.LOG_DISTANCE, constants.RELATIVE_BEARING]
        return BayesianDigitalTwin([cell_data], x_columns, ["avg_rsrp"], **kwargs), len(x_columns)

    def test_ard_gives_one_lengthscale_per_feature_by_default(self):
        twin, num_features = self._twin()
        lengthscale = twin.model.covar_module.base_kernel.lengthscale
        self.assertEqual(lengthscale.shape[-1], num_features)

    def test_ard_can_be_disabled(self):
        twin, _ = self._twin(ard=False)
        lengthscale = twin.model.covar_module.base_kernel.lengthscale
        self.assertEqual(lengthscale.shape[-1], 1)


def _smooth_cell_frame(n, rng):
    """Near-noiseless smooth Rx field over two features, like the MRO training data."""
    a, b = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    return pd.DataFrame({"cell_id": "Cell1", "a": a, "b": b, "y": -80 + 0.3 * np.sin(3 * a) + 0.2 * np.cos(2 * b)})


class TestFantasyUpdate(unittest.TestCase):
    """update_trained_gpmodel must give the exact GP posterior above 800 training rows, where
    gpytorch's predictions cache a float32 Lanczos inverse instead of a Cholesky one."""

    # trained MRO twins land at roughly these hyperparameters
    OUTPUTSCALE, LENGTHSCALE, NOISE = 3.6, 0.6, 0.005

    def setUp(self):
        torch.manual_seed(0)
        self.rng = np.random.default_rng(0)
        self.train = _smooth_cell_frame(2000, self.rng)
        self.test = _smooth_cell_frame(100, self.rng)
        self.twin = BayesianDigitalTwin([self.train], ["a", "b"], ["y"])
        self.twin.model.covar_module.outputscale = self.OUTPUTSCALE
        self.twin.model.covar_module.base_kernel.lengthscale = self.LENGTHSCALE
        self.twin.model.likelihood.noise = self.NOISE

    def _exact_posterior_mean(self, observed):
        """Exact RBF-GP posterior mean in float64 numpy, normalized with the initial training stats."""
        x_min, x_max = self.train[["a", "b"]].min().values, self.train[["a", "b"]].max().values
        y_mean, y_std = self.train["y"].mean(), self.train["y"].std()
        outputscale = self.twin.model.covar_module.outputscale.item()
        lengthscale = self.twin.model.covar_module.base_kernel.lengthscale.flatten()[0].item()
        noise = self.twin.model.likelihood.noise.item()

        def scaled(df):
            return (df[["a", "b"]].values - x_min) / (x_max - x_min) / lengthscale

        def rbf(p, q):
            return outputscale * np.exp(-0.5 * ((p[:, None, :] - q[None, :, :]) ** 2).sum(-1))

        x_obs, x_test = scaled(observed), scaled(self.test)
        y_obs = (observed["y"].values - y_mean) / y_std
        k_obs = rbf(x_obs, x_obs) + noise * np.eye(len(x_obs))
        mean = rbf(x_test, x_obs) @ np.linalg.solve(k_obs, y_obs)
        return mean * y_std + y_mean

    def _predicted_mean(self):
        means, _ = self.twin.predict_distributed_gpmodel([self.test.copy()])
        return np.ravel(means)

    def test_update_after_predict_matches_exact_posterior(self):
        self._predicted_mean()  # MRO predicts (solve) before it updates, which builds the caches
        new = _smooth_cell_frame(500, self.rng)

        self.twin.update_trained_gpmodel([new])

        expected = self._exact_posterior_mean(pd.concat([self.train, new]))
        np.testing.assert_allclose(self._predicted_mean(), expected, atol=1e-3)

    def test_chained_updates_match_exact_posterior(self):
        observed = self.train
        for _ in range(3):
            self._predicted_mean()
            new = _smooth_cell_frame(500, self.rng)
            self.twin.update_trained_gpmodel([new])
            observed = pd.concat([observed, new])

            np.testing.assert_allclose(self._predicted_mean(), self._exact_posterior_mean(observed), atol=1e-3)

    def test_update_without_prior_predict_matches_exact_posterior(self):
        new = _smooth_cell_frame(500, self.rng)

        self.twin.update_trained_gpmodel([new])

        expected = self._exact_posterior_mean(pd.concat([self.train, new]))
        np.testing.assert_allclose(self._predicted_mean(), expected, atol=1e-3)

    def test_update_after_save_and_reload_matches_exact_posterior(self):
        self._predicted_mean()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "model_map.pickle")
            BayesianDigitalTwin.save_model_map_to_pickle(path, {"Cell1": self.twin})
            self.twin = BayesianDigitalTwin.load_model_map_from_pickle(path)["Cell1"]
        new = _smooth_cell_frame(500, self.rng)

        self.twin.update_trained_gpmodel([new])

        expected = self._exact_posterior_mean(pd.concat([self.train, new]))
        np.testing.assert_allclose(self._predicted_mean(), expected, atol=1e-3)


class TestSaveModelMap(unittest.TestCase):
    """save_model_map_to_pickle must leave gpytorch's posterior caches out of the file (tens of MB
    per cell after a prediction or a fantasy update) without dropping them from memory."""

    MAX_FILE_BYTES = 1_000_000  # ~40 KB without the caches, ~20 MB with them

    def setUp(self):
        torch.manual_seed(0)
        self.rng = np.random.default_rng(0)
        self.test = _smooth_cell_frame(100, self.rng)
        self.twin = BayesianDigitalTwin([_smooth_cell_frame(2000, self.rng)], ["a", "b"], ["y"])
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.path = os.path.join(tmp_dir.name, "model_map.pickle")

    def _predicted_mean(self, twin):
        means, _ = twin.predict_distributed_gpmodel([self.test.copy()])
        return np.ravel(means)

    def _save_and_reload(self):
        BayesianDigitalTwin.save_model_map_to_pickle(self.path, {"Cell1": self.twin})
        return BayesianDigitalTwin.load_model_map_from_pickle(self.path)["Cell1"]

    def test_file_excludes_caches_after_predict(self):
        self._predicted_mean(self.twin)

        self._save_and_reload()

        self.assertLess(os.path.getsize(self.path), self.MAX_FILE_BYTES)

    def test_file_excludes_caches_after_update(self):
        self.twin.update_trained_gpmodel([_smooth_cell_frame(500, self.rng)])

        self._save_and_reload()

        self.assertLess(os.path.getsize(self.path), self.MAX_FILE_BYTES)

    def test_in_memory_caches_survive_save(self):
        self._predicted_mean(self.twin)
        caches = self.twin.model.prediction_strategy

        self._save_and_reload()

        self.assertIs(self.twin.model.prediction_strategy, caches)

    def test_reloaded_twin_predicts_like_original(self):
        expected = self._predicted_mean(self.twin)

        reloaded = self._save_and_reload()

        np.testing.assert_allclose(self._predicted_mean(reloaded), expected, atol=1e-4)
