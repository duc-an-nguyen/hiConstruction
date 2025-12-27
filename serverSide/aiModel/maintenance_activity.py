
# ===== Project / Utility =====
from model.tepSystemModel   import SystemModel
from preProcessingDataTEP   import PreProcessData

# ===== Keras / TensorFlow =====
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ===== Data / Math =====
import numpy as np
import pandas as pd
import math
import random
import json
import pickle
import copy

# ===== Visualization =====
import matplotlib.pyplot    as plt
from prettytable            import PrettyTable

# ===== Python Standard Library =====
import os
import warnings
import logging
from collections import namedtuple
from typing import Dict, List, Tuple, Any, Optional
from    utility.logger      import  aiLogger

# Suppress verbose logging from preProcessingDataTEP.getTensorData
logging.getLogger().setLevel(logging.WARNING)


# ==============================
# Config and data preprocessing
# ==============================
class PreBuildAI():
    def __init__(self):

        # ---------------------------
        # Loading config file
        # ---------------------------
        with open('utility/modelConfig.json', 'r') as configFile:
            configData = json.load(configFile)
        config = configData["tep"]  

        # ---------------------------
        # Set debug level and 
        # ---------------------------
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
        warnings.filterwarnings('ignore') 
        
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
        np.random.seed(1)
        random.seed(1)
        tf.random.set_seed(1)

        outputDir               = os.path.expanduser(config["dataDir"])
        totalTraject            = 100
        normalizeHI             = True
        showRedundantShutdown   = False


        self.preData             = PreProcessData(config=config)
        aiLogger.warning("PreBuildAI run again !!!")
        rawData             = self.preData.loadDataFromH5Format(dir              = outputDir, 
                                                                name             = "data/teps.h5",
                                                                case             = "all", 
                                                                reduceSampleRate = True, 
                                                                removeHigherThr  = False)

        withRULData         = self.preData.addRULAndEnvironment(rawData)
        # withoutInitRawData  = preData.removeInitState(inData= withRULData)



        # ---------------------------
        # For data
        cleanData           = self.preData.cleanData(rawSignal           =   withRULData,
                                                columnName          =   self.preData.allWithoutLabel,
                                                getMovingAverage    =   False,
                                                flipToDown          =   False)
        # ---------------------------
        # For label, flip only
        cleanData           = self.preData.cleanData(rawSignal           =   cleanData,
                                                columnName          =   self.preData.degraLiq,
                                                getMovingAverage    =   False,
                                                flipToDown          =   False)

        self.originalData        = cleanData.copy()
        self.norData             = self.preData.normalizeDataEachModes(rawData    = cleanData,
                                                            headline   =   self.preData.headline["allWithoutLabelNoBias"] + ["rul"])    # Normalize the data


        config["component"] = "Sys"
        self.vaeSys_global_best  = SystemModel(config = config)
        self.vaeSys_global_best.build(input_shape=[(None, config["seqLength"], config["inReac"]), (None, 1, config["inReac"]), (None, 1)])
        self.vaeSys_global_best.summary()



# ============================================================
# A. env(2/3/4) client construction + SystemModel evaluation + HI trajectories
#    (A1/A2/A3 can be pre-run separately)
# ============================================================

# ---------------------- A1. client construction ----------------------
class ClientConstruction():
    def __init__(self):
        pass
    def _to_int(self,x: Any) -> Any:
        try:
            return int(x)
        except (ValueError, TypeError):
            return x


    def _ensure_env_maps_to_real_ids(self,data: Any) -> Dict[Any, Any]:
        """
        Convert any data format to {env_id: df}, where env_id is the real value from df['environmentID'] (2/3/4)
        """
        env_map: Dict[Any, Any] = {}

        def _ingest_df(df):
            assert "environmentID" in df.columns, "Column 'environmentID' not found."
            for eid, g in df.groupby("environmentID", sort=False):
                env_map[self._to_int(eid)] = g

        if hasattr(data, "keys") and callable(getattr(data, "keys")):
            for k in list(data.keys()):
                _ingest_df(data[k])
            return env_map

        if isinstance(data, (list, tuple)):
            for i in range(len(data)):
                _ingest_df(data[i])
            return env_map

        _ingest_df(data)
        return env_map


    def _get_sequences_for_one_trajectory(self,
        env_df_normalized,
        env_df_original,
        preData,
        getAtIdx: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Use preData.getTensorData for 1 trajectory (1-based getAtIdx),
        Returns:
            seqXAllSensor: (n_seq, seq_len, n_features)   - normalized sensors
            seqXRul:       (n_seq, 1)                     - original RUL at last step
            seqXEnv:       (n_seq, seq_len, env_dim)
            seqTime:       (n_seq, seq_len, time_dim)
        """
        trajectID = env_df_normalized["Id"]
        allSensor = env_df_normalized[preData.headline["allWithoutLabelNoBias"]]
        env_col = env_df_normalized["environmentID"]
        time_col = env_df_normalized["Time"]

        # Original (unnormalized) RUL
        rul_original = env_df_original["rul"]

        # 1) Normalized features
        seqXAllSensor, _, seqXEnv, seqTime = preData.getTensorData(
            dataList=[allSensor, allSensor.iloc[:, :1], env_col, time_col],
            trajectId=trajectID,
            shuffle=False,
            getAtIdx=getAtIdx,
            seqLength=preData.config["seqLength"],
            overlap=preData.config["overLap"],
        )

        # 2) Use original RUL, take last timestep of each window
        _, seqXRul_full, _, _ = preData.getTensorData(
            dataList=[allSensor.iloc[:, :1], rul_original, env_col, time_col],
            trajectId=trajectID,
            shuffle=False,
            getAtIdx=getAtIdx,
            seqLength=preData.config["seqLength"],
            overlap=preData.config["overLap"],
        )
        seqXRul = seqXRul_full[:, -1, :]  # (n_seq, 1)

        return seqXAllSensor, seqXRul, seqXEnv, seqTime


    def _stack(self,arrs: List[np.ndarray]) -> np.ndarray:
        if not arrs:
            return np.empty((0,))
        return np.concatenate(arrs, axis=0)


    def _split_indices(self,
        n_items: int,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[List[int], List[int], List[int]]:
        rng = np.random.default_rng(seed)
        idx = np.arange(n_items)
        rng.shuffle(idx)

        n_train = int(round(train_ratio * n_items))
        n_val = int(round(val_ratio * n_items))

        train_idx = idx[:n_train].tolist()
        val_idx = idx[n_train: n_train + n_val].tolist()
        test_idx = idx[n_train + n_val:].tolist()
        return train_idx, val_idx, test_idx


    def _federated_split_env(self,
        env_df_normalized,
        env_df_original,
        preData,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
        ) -> Dict[str, Dict[str, Any]]:
        """
        Within a single env, split by trajectory into train/val/test.
        Returns each split as:
            {
            "X": [X_all, X_time, X_env],
            "y": y_TRUE_RUL_last_step,
            "meta": {
                "traj_ids":     (N,),
                "env_traj_ids": (N,),
                "traj_lengths": (num_traj,),
                "num_traj":     int,
            }
            }
        """
        unique_ids = env_df_normalized["Id"].unique().tolist()
        total_traj = len(unique_ids)

        if total_traj == 0:
            empty_X = [np.empty((0,)), np.empty((0,)), np.empty((0,))]
            empty_y = np.empty((0,))
            empty_meta = {
                "traj_ids": np.empty((0,), dtype=np.int32),
                "env_traj_ids": np.empty((0,), dtype=np.int32),
                "traj_lengths": np.empty((0,), dtype=np.int32),
                "num_traj": 0,
            }
            empty = {"X": empty_X, "y": empty_y, "meta": empty_meta}
            return {"train": empty, "val": empty, "test": empty}

        X_all_list: List[np.ndarray] = []
        X_time_list: List[np.ndarray] = []
        X_env_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []

        for getAtIdx in range(1, total_traj + 1):
            seqXAllSensor, seqXRul, seqXEnv, seqTime = self._get_sequences_for_one_trajectory(
                env_df_normalized=env_df_normalized,
                env_df_original=env_df_original,
                preData=preData,
                getAtIdx=getAtIdx,
            )
            X_all_list.append(seqXAllSensor)
            X_time_list.append(seqTime)
            X_env_list.append(seqXEnv)
            y_list.append(seqXRul)

        train_ids, val_ids, test_ids = self._split_indices(
            n_items=total_traj, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
        )

        def _concat(id_list: List[int]) -> Tuple[List[np.ndarray], np.ndarray, Dict[str, Any]]:
            if not id_list:
                X = [np.empty((0,)), np.empty((0,)), np.empty((0,))]
                y = np.empty((0,))
                meta = {
                    "traj_ids": np.empty((0,), dtype=np.int32),
                    "env_traj_ids": np.empty((0,), dtype=np.int32),
                    "traj_lengths": np.empty((0,), dtype=np.int32),
                    "num_traj": 0,
                }
                return X, y, meta

            X_all_parts, X_time_parts, X_env_parts, y_parts = [], [], [], []
            traj_ids: List[int] = []
            env_traj_ids: List[int] = []
            traj_lengths: List[int] = []

            for local_tid, env_tid in enumerate(id_list):
                Xa_i = X_all_list[env_tid]
                Xt_i = X_time_list[env_tid]
                Xe_i = X_env_list[env_tid]
                y_i = y_list[env_tid]
                n_i = Xa_i.shape[0]

                X_all_parts.append(Xa_i)
                X_time_parts.append(Xt_i)
                X_env_parts.append(Xe_i)
                y_parts.append(y_i)

                traj_ids.extend([local_tid] * n_i)
                env_traj_ids.extend([env_tid] * n_i)
                traj_lengths.append(n_i)

            X_all = self._stack(X_all_parts)
            X_time = self._stack(X_time_parts)
            X_env = self._stack(X_env_parts)
            y = self._stack(y_parts)

            traj_ids_arr = np.asarray(traj_ids, dtype=np.int32)
            env_traj_ids_arr = np.asarray(env_traj_ids, dtype=np.int32)
            traj_lengths_arr = np.asarray(traj_lengths, dtype=np.int32)

            meta = {
                "traj_ids": traj_ids_arr,
                "env_traj_ids": env_traj_ids_arr,
                "traj_lengths": traj_lengths_arr,
                "num_traj": len(id_list),
            }
            return [X_all, X_time, X_env], y, meta

        X_train, y_train, meta_train = _concat(train_ids)
        X_val, y_val, meta_val = _concat(val_ids)
        X_test, y_test, meta_test = _concat(test_ids)

        return {
            "train": {"X": X_train, "y": y_train, "meta": meta_train},
            "val": {"X": X_val, "y": y_val, "meta": meta_val},
            "test": {"X": X_test, "y": y_test, "meta": meta_test},
        }


    def build_env_clients_2_3_4(self,
        preData,
        norData,
        originalData,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        """
        Returns 9 client variables:
            train_client_2, val_client_2, test_client_2,
            train_client_3, val_client_3, test_client_3,
            train_client_4, val_client_4, test_client_4

        Each client: {"X": [X_all, X_time, X_env], "y": y, "meta": {...}}
        """
        env_map_normalized = self._ensure_env_maps_to_real_ids(norData)
        env_map_original = self._ensure_env_maps_to_real_ids(originalData)

        required_envs = [2, 3, 4]
        for eid in required_envs:
            assert eid in env_map_normalized, f"Environment ID {eid} not found in normalized data."
            assert eid in env_map_original, f"Environment ID {eid} not found in original data."

        for eid in required_envs:
            df_norm = env_map_normalized[eid]
            df_orig = env_map_original[eid]
            assert "Id" in df_norm.columns
            assert "Time" in df_norm.columns
            assert "environmentID" in df_norm.columns
            assert "rul" in df_orig.columns
            _ = df_norm[preData.headline["allWithoutLabelNoBias"]]

        client2 = self._federated_split_env(
            env_map_normalized[2], env_map_original[2], preData, train_ratio, val_ratio, seed
        )
        client3 = self._federated_split_env(
            env_map_normalized[3], env_map_original[3], preData, train_ratio, val_ratio, seed
        )
        client4 = self._federated_split_env(
            env_map_normalized[4], env_map_original[4], preData, train_ratio, val_ratio, seed
        )

        train_client_2 = client2["train"]
        val_client_2 = client2["val"]
        test_client_2 = client2["test"]

        train_client_3 = client3["train"]
        val_client_3 = client3["val"]
        test_client_3 = client3["test"]

        train_client_4 = client4["train"]
        val_client_4 = client4["val"]
        test_client_4 = client4["test"]

        return (
            train_client_2,
            val_client_2,
            test_client_2,
            train_client_3,
            val_client_3,
            test_client_3,
            train_client_4,
            val_client_4,
            test_client_4,
        )


# ---------------------- A2. SystemModel evaluation ----------------------

    def show_sensor_masks(self,model, preData):
        """
        Print sensor selection masks for each component (Reac/Sep/Strip) in env 2/3/4
        """
        components = [
            ("Reac", model.maskCr1),
            ("Sep", model.maskCr2),
            ("Strip", model.maskCv2),
        ]
        sensor_names = preData.headline["allWithoutLabelNoBias"]

        for comp_name, mask_layer in components:
            binary_mask, probs = mask_layer.getMask()  # (3, num_sensors)
            print(f"\n========== Component: {comp_name} ==========")
            for env_idx in range(binary_mask.shape[0]):
                env_id = env_idx + 2
                bm_env = binary_mask[env_idx]
                pb_env = probs[env_idx]
                num_selected = int(bm_env.numpy().sum())
                print(f"Environment {env_id}: selected sensors = {num_selected}")

                table = PrettyTable()
                table.field_names = ["No", "sensorName", "probability", "binary"]
                for i, s_name in enumerate(sensor_names):
                    table.add_row(
                        [i, s_name, float(pb_env[i].numpy()), int(bm_env[i].numpy())]
                    )
                print(table)


    def evaluate_on_split(self,model, split_dict, env_id, batch_size=256):
        """
        Run SystemModel.getHIandRUL on a split (train/val/test) of one env
        """
        from scipy.stats import spearmanr

        X_all, X_time, X_env = split_dict["X"]
        y_true = split_dict["y"].reshape(-1, 1)
        meta = split_dict.get("meta", None)

        N = X_all.shape[0]
        if N == 0:
            print(f"[Warning] env {env_id} split is empty.")
            res = {
                "HI_comp1": np.empty((0,)),
                "HI_comp2": np.empty((0,)),
                "HI_comp3": np.empty((0,)),
                "HI_sys": np.empty((0,)),
                "RUL_true": np.empty((0,)),
                "RUL_pred": np.empty((0,)),
                "metrics": {"MAE": np.nan, "RMSE": np.nan, "SpearmanR": np.nan},
            }
            if meta is not None:
                res["meta"] = copy.deepcopy(meta)
            return res

        seqCycle_dummy = np.zeros((N, 1), dtype=np.float32)
        dataset = (
            tf.data.Dataset.from_tensor_slices(
                (
                    X_all.astype(np.float32),
                    seqCycle_dummy.astype(np.float32),
                    y_true.astype(np.float32),
                )
            ).batch(batch_size)
        )

        hi1_list, hi2_list, hi3_list = [], [], []
        sys_hi_list, y_pred_list = [], []
        env_scalar = tf.constant(env_id, dtype=tf.int32)

        for seqData_b, seqCycle_b, y_b in dataset:
            zCr1_b, zCr2_b, zCv2_b, sysHi_b, predRUL_b, _ = model.getHIandRUL(
                (seqData_b, seqCycle_b, env_scalar)
            )
            hi1_list.append(zCr1_b.numpy())
            hi2_list.append(zCr2_b.numpy())
            hi3_list.append(zCv2_b.numpy())
            sys_hi_list.append(sysHi_b.numpy())
            y_pred_list.append(predRUL_b.numpy())

        HI_comp1 = np.concatenate(hi1_list, axis=0)
        HI_comp2 = np.concatenate(hi2_list, axis=0)
        HI_comp3 = np.concatenate(hi3_list, axis=0)
        HI_sys = np.concatenate(sys_hi_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)
        y_true_flat = y_true[: y_pred.shape[0]]

        mae = np.mean(np.abs(y_pred - y_true_flat))
        rmse = np.sqrt(np.mean((y_pred - y_true_flat) ** 2))
        try:
            rho, _ = spearmanr(y_true_flat.squeeze(), y_pred.squeeze())
        except Exception:
            rho = np.nan

        print(f"Env {env_id}  ->  MAE = {mae:.4f}, RMSE = {rmse:.4f}, Spearman = {rho:.4f}")

        res = {
            "HI_comp1": HI_comp1,
            "HI_comp2": HI_comp2,
            "HI_comp3": HI_comp3,
            "HI_sys": HI_sys,
            "RUL_true": y_true_flat,
            "RUL_pred": y_pred,
            "metrics": {"MAE": float(mae), "RMSE": float(rmse), "SpearmanR": float(rho)},
        }
        if meta is not None:
            res["meta"] = copy.deepcopy(meta)
        return res


    def evaluate_system_model_on_clients(self,model, clients_by_env, batch_size=256):
        """
        clients_by_env = {
        2: {"train":..., "val":..., "test":...},
        3: {...},
        4: {...},
        }
        """
        results = {}
        for env_id, splits_dict in clients_by_env.items():
            print(f"\n================ Env {env_id} ================")
            results[env_id] = {}
            for split_name in ["train", "val", "test"]:
                client_split = splits_dict.get(split_name, None)
                if client_split is None:
                    print(f"[Skip] env {env_id}, split {split_name} is None.")
                    continue
                print(f"\n--- Evaluating split: {split_name} ---")
                res = self.evaluate_on_split(
                    model=model,
                    split_dict=client_split,
                    env_id=env_id,
                    batch_size=batch_size,
                )
                results[env_id][split_name] = res
        return results


# ---------------------- A3. HI trajectories & split by trajectory ----------------------


    def build_hi_traj_from_eval_result(self,
        res,
        alpha: float = 0.1,
        w_sys: float = 0.3,
    ):
        """
        Convert evaluate_on_split result to full HI trajectories:
            hi_traj_all : (N, 3)
            sys_hi_traj : (N,)
            rul         : (N,) or None
        """
        z1 = np.asarray(res["HI_comp1"], dtype=np.float32)
        z2 = np.asarray(res["HI_comp2"], dtype=np.float32)
        z3 = np.asarray(res["HI_comp3"], dtype=np.float32)
        sys_raw = np.asarray(res["HI_sys"], dtype=np.float32).reshape(-1)

        rul_arr = res.get("RUL_pred", None)
        if rul_arr is not None:
            rul = np.asarray(rul_arr, dtype=np.float32)
            if rul.ndim > 1:
                rul = rul.reshape(-1)
        else:
            rul = None

        T = z1.shape[0]
        assert z2.shape[0] == T and z3.shape[0] == T and sys_raw.shape[0] == T

        def _latent_to_scalar(z_vec):
            l2 = np.linalg.norm(z_vec, ord=2, axis=-1)
            return np.exp(-alpha * l2)

        h1_c = _latent_to_scalar(z1)
        h2_c = _latent_to_scalar(z2)
        h3_c = _latent_to_scalar(z3)

        h_sys = np.tanh(sys_raw)
        h_sys = 0.5 * (h_sys + 1.0)

        w_comp = 1.0 - w_sys
        h1 = w_comp * h1_c + w_sys * h_sys
        h2 = w_comp * h2_c + w_sys * h_sys
        h3 = w_comp * h3_c + w_sys * h_sys

        hi_traj = np.stack([h1, h2, h3], axis=-1)
        hi_traj = np.clip(hi_traj, 0.0, 1.2)

        sys_hi_traj = h_sys
        return hi_traj, sys_hi_traj, rul


    def build_hi_trajs_per_trajectory(self,
        res,
        alpha: float = 0.1,
        w_sys: float = 0.3,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Optional[np.ndarray]]]:
        """
        Use meta['traj_lengths'] to split HI by trajectory:
            hi_trajs     : List[(T_k, 3)]
            sys_hi_trajs : List[(T_k,)]
            rul_trajs    : List[(T_k,) or None]
        """
        hi_all, sys_all, rul_all = self.build_hi_traj_from_eval_result(res, alpha, w_sys)

        meta = res.get("meta", None)
        if meta is None:
            raise ValueError(
                "res missing 'meta', cannot split HI by trajectory. "
                "Ensure using new build_env_clients_2_3_4 + evaluate_system_model_on_clients."
            )

        traj_lengths = meta.get("traj_lengths", None)
        num_traj = meta.get("num_traj", None)
        if traj_lengths is None or num_traj is None:
            raise ValueError("meta missing 'traj_lengths' or 'num_traj'.")

        traj_lengths = np.asarray(traj_lengths, dtype=np.int64)
        num_traj = int(num_traj)
        if traj_lengths.shape[0] != num_traj:
            raise ValueError("traj_lengths length does not match num_traj.")

        total_len = int(traj_lengths.sum())
        if total_len != hi_all.shape[0]:
            raise ValueError(
                f"traj_lengths.sum()={total_len} does not match HI total length={hi_all.shape[0]}."
            )

        hi_trajs, sys_hi_trajs, rul_trajs = [], [], []
        offset = 0
        for k in range(num_traj):
            L = int(traj_lengths[k])
            if L <= 0:
                continue
            hi_k = hi_all[offset: offset + L]
            sys_k = sys_all[offset: offset + L]
            if rul_all is not None:
                rul_k = rul_all[offset: offset + L]
            else:
                rul_k = None
            hi_trajs.append(hi_k)
            sys_hi_trajs.append(sys_k)
            rul_trajs.append(rul_k)
            offset += L

        return hi_trajs, sys_hi_trajs, rul_trajs


    def auto_select_trigger_step(self,
        hi_traj: np.ndarray,
        sys_hi_traj: np.ndarray,
        request_threshold: float,
        failure_threshold: float,
        prefer_index: Optional[int] = None,
    ) -> int:
        """
        Auto-select trigger point on a single trajectory's HI:
        - Prefer first min(HI_comp, HI_sys) <= request_threshold
        - Otherwise take the point with lowest average HI
        """
        hi_traj = np.asarray(hi_traj, dtype=np.float32)
        sys_hi_traj = np.asarray(sys_hi_traj, dtype=np.float32).reshape(-1)
        T = hi_traj.shape[0]

        if prefer_index is not None and 0 <= prefer_index < T:
            return int(prefer_index)

        comp_min = hi_traj.min(axis=1)
        cond = (comp_min <= request_threshold) | (sys_hi_traj <= request_threshold)
        idx = np.where(cond)[0]
        if idx.size > 0:
            return int(idx[0])

        avg_hi = hi_traj.mean(axis=1)
        return int(np.argmin(avg_hi))


    def precompute_hi_repository(self,
        preData,
        norData,
        originalData,
        vaeSys_global_best,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        seed: int = 42,
        batch_size: int = 256,
        alpha: float = 0.1,
        w_sys: float = 0.3,
    ):
        """
        Run A1/A2/A3 at once, output hi_repo for subsequent RL use.

        Returns:
            {
            "clients_by_env": {2:{...},3:{...},4:{...}},
            "eval_results":   {2:{split:res}, ...},
            "hi_repo": {
                env_id: {
                    split_name: {
                        "hi":      List[np.ndarray],  # each trajectory's (T_k,3)
                        "sys_hi":  List[np.ndarray],  # (T_k,)
                        "rul":     List[Optional[np.ndarray]]
                    },
                    ...
                },
                ...
            }
            }
        """
        print(">>> [HI Precompute] Building env clients...")
        (
            train_client_2,
            val_client_2,
            test_client_2,
            train_client_3,
            val_client_3,
            test_client_3,
            train_client_4,
            val_client_4,
            test_client_4,
        ) = self.build_env_clients_2_3_4(
            preData=preData,
            norData=norData,
            originalData=originalData,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

        clients_by_env = {
            2: {"train": train_client_2, "val": val_client_2, "test": test_client_2},
            3: {"train": train_client_3, "val": val_client_3, "test": test_client_3},
            4: {"train": train_client_4, "val": val_client_4, "test": test_client_4},
        }

        print(">>> [HI Precompute] Printing sensor masks (for diagnostics)...")
        self.show_sensor_masks(vaeSys_global_best, preData)

        print(">>> [HI Precompute] SystemModel evaluating each env/split ...")
        all_results = self.evaluate_system_model_on_clients(
            model=vaeSys_global_best,
            clients_by_env=clients_by_env,
            batch_size=batch_size,
        )

        print(">>> [HI Precompute] Splitting HI by trajectory...")
        hi_repo: Dict[int, Dict[str, Any]] = {}
        for env_id in [2, 3, 4]:
            hi_repo[env_id] = {}
            for split_name in ["train", "val", "test"]:
                res = all_results[env_id][split_name]
                hi_trajs, sys_hi_trajs, rul_trajs = self.build_hi_trajs_per_trajectory(
                    res, alpha=alpha, w_sys=w_sys
                )
                hi_repo[env_id][split_name] = {
                    "hi": hi_trajs,
                    "sys_hi": sys_hi_trajs,
                    "rul": rul_trajs,
                }
                print(
                    f"[HI Precompute] env={env_id}, split={split_name}, "
                    f"num_traj={len(hi_trajs)}"
                )

        return {
            "clients_by_env": clients_by_env,
            "eval_results": all_results,
            "hi_repo": hi_repo,
        }


# ============================================================
# B. KAN process model + extended plant maintenance/logistics environment + Planner + Scenario
# ============================================================

# ---------------------- B1. Maintenance strategies & business profiles ----------------------



BUSINESS_PROFILES = {
    "aggressive": {"w_profit": 1.0, "w_maint": 0.4, "w_fail": 0.4, "w_stockout": 0.3, "w_hi": 0.0},
    "balanced": {
        "w_profit": 1.0,
        "w_maint": 0.7,
        "w_fail": 0.9,
        "w_stockout": 0.6,
        "w_hi": 0.3,
    },
    "conservative": {
        "w_profit": 0.9,
        "w_maint": 0.9,
        "w_fail": 1.3,
        "w_stockout": 1.0,
        "w_hi": 0.8,
    },
}

# ---------------------- B2. KAN process model (TensorFlow) ----------------------


class LinearOp(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.a = self.add_weight(
            name="a",
            shape=(1,),
            initializer="ones",
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, h):
        return self.a * h + self.b



class QuadraticOp(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.a = self.add_weight(
            name="a",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(1,),
            initializer="ones",
            trainable=True,
        )
        self.c = self.add_weight(
            name="c",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, h):
        return self.a * h ** 2 + self.b * h + self.c



class LogOp(keras.layers.Layer):
    def __init__(self, eps=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(1,),
            initializer="ones",
            trainable=True,
        )

    def call(self, h):
        return self.scale * tf.math.log(tf.abs(h) + self.eps)


class SigmoidOp(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(1,),
            initializer="ones",
            trainable=True,
        )
        self.slope = self.add_weight(
            name="slope",
            shape=(1,),
            initializer="ones",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, h):
        return self.scale * tf.sigmoid(self.slope * h + self.bias)



class SinOp(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(1,),
            initializer="ones",
            trainable=True,
        )
        self.freq = self.add_weight(
            name="freq",
            shape=(1,),
            initializer="ones",
            trainable=True,
        )
        self.phase = self.add_weight(
            name="phase",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, h):
        return self.scale * tf.sin(self.freq * h + self.phase)



class KANOperatorBlock(keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def build(self, input_shape):
        self.encoder = layers.Dense(self.hidden_dim, activation="tanh")
        self.decoder = layers.Dense(self.output_dim)
        self.ops = [LinearOp(), QuadraticOp(), LogOp(), SigmoidOp(), SinOp()]
        self.n_ops = len(self.ops)
        self.gate = keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dense(self.n_ops),
        ])
        for op in self.ops:
            op.build((None, self.hidden_dim))

    def call(self, x):
        h = self.encoder(x)
        op_outs = [op(h) for op in self.ops]
        op_stack = tf.stack(op_outs, axis=-1)  # (B, H, K)
        logits = self.gate(x)
        weights = tf.nn.softmax(logits, axis=-1)  # (B, K)
        combined = tf.reduce_sum(op_stack * tf.expand_dims(weights, 1), axis=-1)  # (B, H)
        y = self.decoder(combined)
        return y, weights, op_stack


class NextStepKANProcessModel(keras.Model):
    """
    Original n_channels + 3-dim HI => next step n_channels
    """

    def __init__(
        self,
        n_channels: int,
        n_components: int = 3,
        hidden_dim: int = 64,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_channels = int(n_channels)
        self.n_components = int(n_components)
        self.hidden_dim = int(hidden_dim)
        input_dim = self.n_channels + self.n_components
        output_dim = self.n_channels
        self.block = KANOperatorBlock(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
        )

    def call(self, x_raw_hi):
        return self.block(x_raw_hi)[0]

    def predict_next(self, raw_t, hi_t):
        """
        Single-step prediction of next raw channels, used for PlantMaintenanceEnvV3's "world model" evolution.
        """
        raw_t = np.asarray(raw_t, dtype=np.float32)
        hi_t = np.asarray(hi_t, dtype=np.float32)
        if raw_t.ndim == 1:
            raw_t = raw_t[None, :]
        if hi_t.ndim == 1:
            hi_t = hi_t[None, :]
        x = np.concatenate([raw_t, hi_t], axis=-1)
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        y, _, _ = self.block(x_tensor)
        y_np = y.numpy()
        if y_np.shape[0] == 1:
            return y_np[0]
        return y_np

    def get_config(self):
        return {
            "n_channels": self.n_channels,
            "n_components": self.n_components,
            "hidden_dim": self.hidden_dim,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ---------------------- B3. Extended plant maintenance + logistics environment ----------------------

class PlantMaintenanceEnvV3:
    """
    Extended environment:
      - Original 3-component HI degradation + PM/CM + spare inventory/reorder/lead time
      - Production side: Stripper -> Separator -> Compressor -> finished goods
      - Logistics side:
          * Finished goods inventory fg_inventory
          * Random demand ~ N(mean,std)+, limited by ship_capacity
          * Backlog penalty + FG holding cost + shipping cost
      - RL learns: how to maintain multiple times within limited maintenance budget
      - Reward is designed to be POSITIVE for good performance (profit-based)
    """

    def __init__(self,
                config      =None,
                process_model: Optional[NextStepKANProcessModel] = None,
                seed: int = 0,):
        

        self.MAINT_STRATEGIES = {"Perfect Maintenance": {
                                "recover_ratio": 0.90,
                                "slope_factor": 0.50,
                                "min_dHI": 0.10,
                                "noise_mult": 0.30,},
                            "General Maintenance": {
                                "recover_ratio": 0.45,
                                "slope_factor": 0.85,
                                "min_dHI": 0.05,
                                "noise_mult": 0.55,
                            },
                            "Minor Maintenance": {
                                "recover_ratio": 0.12,
                                "slope_factor": 1.15,
                                "min_dHI": 0.00,
                                "noise_mult": 1.05,
                            },}


        if config is None:
            config = {}
        self._cfg = copy.deepcopy(config)
        self.rng = np.random.RandomState(seed)

        self.n_components = 3
        self.comp_names = ["Stripper", "Separator", "Compressor"]
        self.n_channels = int(config.get("n_channels", 45))
        self.max_steps = int(config.get("max_steps", 365))

        # HI degradation parameters
        self.base_degrade_rates = np.array(
            config.get("base_degrade_rates", [0.0020, 0.0015, 0.0018]),
            dtype=np.float32,
        )
        self.base_noise_std = float(config.get("random_degrade_std", 0.0003))
        self.failure_threshold = float(config.get("failure_threshold", 0.2))
        self.request_threshold = float(config.get("request_threshold", 0.4))
        self.hi_reset = float(config.get("hi_reset", 1.0))

        self.degrade_rates = self.base_degrade_rates.copy()
        self.noise_std = np.full(
            self.n_components, self.base_noise_std, dtype=np.float32
        )

        # Capacity settings (TE-style continuous production)
        self.capacity_per_step = float(config.get("capacity_per_step", 100.0))

        # Sales price & variable cost (per unit) - adjusted for positive profit
        self.price_per_unit = float(config.get("price_per_unit", 15.0))
        self.variable_cost_per_unit = float(
            config.get("variable_cost_per_unit", 3.0)
        )

        # Intermediate buffer tank (process buffer)
        self.buffer_capacity = float(
            config.get("buffer_capacity", self.capacity_per_step * 10.0)
        )
        self.buffer_init = float(
            config.get("buffer_init", self.capacity_per_step * 5.0)
        )

        # Logistics: FG inventory & customer demand & shipping capacity & costs
        self.demand_mean = float(
            config.get("demand_mean", self.capacity_per_step * 0.8)
        )
        self.demand_std = float(config.get("demand_std", self.capacity_per_step * 0.15))
        self.ship_capacity = float(
            config.get("ship_capacity", self.capacity_per_step * 1.2)
        )
        self.holding_cost_per_unit = float(
            config.get("holding_cost_per_unit", 0.1)
        )
        self.backlog_penalty_per_unit = float(
            config.get("backlog_penalty_per_unit", 0.5)
        )
        self.ship_cost_per_unit = float(
            config.get("ship_cost_per_unit", 0.05)
        )
        self.fg_init = float(config.get("fg_init", 200.0))

        # PM/CM costs & downtime - reduced for more positive rewards
        self.pm_costs = np.array(
            config.get("pm_costs", [100.0, 150.0, 200.0]), dtype=np.float32
        )
        self.cm_costs = np.array(
            config.get("cm_costs", [500.0, 600.0, 700.0]), dtype=np.float32
        )
        self.pm_downtime_base = int(config.get("pm_downtime_base", 1))
        self.cm_downtime = int(config.get("cm_downtime", 3))

        # Maintenance level mapping
        self.maint_levels = ["minor", "medium", "major"]
        self.level_to_strategy = {
            "minor": "Minor Maintenance",
            "medium": "General Maintenance",
            "major": "Perfect Maintenance",
        }
        self.level_cost_mult = {"minor": 0.5, "medium": 1.0, "major": 1.5}
        self.level_downtime_mult = {"minor": 0.5, "medium": 1.0, "major": 1.5}

        # Spare inventory + reorder policy (reorder point/quantity) + logistics
        self.initial_spares = np.array(
            config.get("initial_spares", [5, 5, 5]), dtype=np.int32
        )
        self.max_spares = float(config.get("max_spares", 10.0))
        self.reorder_point = np.array(
            config.get("reorder_point", [2, 2, 2]), dtype=np.int32
        )
        self.reorder_qty = np.array(
            config.get("reorder_qty", [3, 3, 3]), dtype=np.int32
        )
        self.lead_time = int(config.get("lead_time", 7))
        self.stockout_extra_downtime = int(
            config.get("stockout_extra_downtime", 3)
        )
        self.stockout_extra_cost = float(
            config.get("stockout_extra_cost", 200.0)
        )
        self.spare_holding_cost_per_unit = float(
            config.get("spare_holding_cost_per_unit", 0.2)
        )

        # Business profile
        profile_name = config.get("business_profile", "balanced")
        if profile_name not in BUSINESS_PROFILES:
            raise ValueError(f"Unknown business profile: {profile_name}")
        self.business_profile = profile_name
        self.vf_weights = BUSINESS_PROFILES[profile_name].copy()

        # Limit maintenance times per episode & extra penalty (drives RL to learn "maintenance count")
        self.max_maint_times = config.get("max_maint_times", None)
        self.maint_extra_penalty = float(config.get("maint_extra_penalty", 0.0))

        # Actions: 0=no maintenance, others = (comp, level)
        self.n_actions = 1 + self.n_components * len(self.maint_levels)

        self.process_model = process_model

        # Rough estimate of max RUL steps
        self.max_rul_steps = (1.0 - self.failure_threshold) / np.min(
            self.base_degrade_rates
        )
        self.reset()

    def set_business_profile(self, profile_name):
        if profile_name not in BUSINESS_PROFILES:
            raise ValueError(f"Unknown business profile: {profile_name}")
        self.business_profile = profile_name
        self.vf_weights = BUSINESS_PROFILES[profile_name].copy()

    def reset(self):
        self.t = 0
        self.health = np.ones(self.n_components, dtype=np.float32)
        self.age = np.zeros(self.n_components, dtype=np.float32)
        self.request_flags = np.zeros(self.n_components, dtype=bool)

        # Maintenance times used
        self.maint_times_used = 0

        # Spare inventory/ordering
        self.spares = self.initial_spares.copy()
        self.on_order_eta = -np.ones(self.n_components, dtype=np.int32)
        self.on_order_qty = np.zeros(self.n_components, dtype=np.int32)

        # Component state
        self.down_remaining = np.zeros(self.n_components, dtype=np.int32)
        self.failed = np.zeros(self.n_components, dtype=bool)

        # Reset degradation parameters to base values
        self.degrade_rates = self.base_degrade_rates.copy()
        self.noise_std = np.full(
            self.n_components, self.base_noise_std, dtype=np.float32
        )

        # Raw channels
        self.raw_x = self.rng.normal(0.0, 1.0, size=(self.n_channels,)).astype(
            np.float32
        )

        # Process buffer tank
        self.buffer_inventory = float(
            np.clip(self.buffer_init, 0.0, self.buffer_capacity)
        )

        # FG inventory + backlog
        self.fg_inventory = float(max(self.fg_init, 0.0))
        self.backlog = 0.0

        self.hi_history = [self.health.copy()]
        return self._get_obs()

    def _get_obs(self):
        raw_norm = np.tanh(self.raw_x)
        hi = self.health.astype(np.float32)
        age_norm = (self.age / float(self.max_steps)).astype(np.float32)
        rul = self._compute_rul()
        rul_norm = (rul / float(self.max_rul_steps)).astype(np.float32)
        spares_norm = (self.spares.astype(np.float32) / self.max_spares).astype(
            np.float32
        )
        req = self.request_flags.astype(np.float32)
        t_norm = np.array([self.t / float(self.max_steps)], dtype=np.float32)

        fg_norm = np.array(
            [self.fg_inventory / (self.capacity_per_step * 10.0 + 1e-6)],
            dtype=np.float32,
        )
        backlog_norm = np.array(
            [self.backlog / (self.demand_mean * 10.0 + 1e-6)],
            dtype=np.float32,
        )

        if self.max_maint_times is not None and self.max_maint_times > 0:
            maint_budget_norm = np.array(
                [self.maint_times_used / float(self.max_maint_times)],
                dtype=np.float32,
            )
        else:
            maint_budget_norm = np.array(
                [self.maint_times_used / float(self.max_steps)],
                dtype=np.float32,
            )

        obs = np.concatenate(
            [
                raw_norm,
                hi,
                age_norm,
                rul_norm,
                spares_norm,
                req,
                t_norm,
                fg_norm,
                backlog_norm,
                maint_budget_norm,
            ],
            axis=0,
        )
        return obs

    def _compute_rul(self):
        degr = np.maximum(self.degrade_rates, 1e-6)
        rem = np.maximum(self.health - self.failure_threshold, 0.0)
        return rem / degr

    def _decode_action(self, action):
        if action == 0:
            return None, None
        idx = action - 1
        comp = idx // len(self.maint_levels)
        level_idx = idx % len(self.maint_levels)
        level = self.maint_levels[level_idx]
        return comp, level

    def _apply_maintenance_effect(self, comp, level, pm_cost, stockout_cost, info):
        # If exceeded maintenance limit, maintenance is blocked (treated as no-op)
        if (
            self.max_maint_times is not None
            and self.maint_times_used >= self.max_maint_times
        ):
            info["pm_blocked"] = True
            return pm_cost, stockout_cost, info

        strategy_name = self.level_to_strategy[level]
        strat = self.MAINT_STRATEGIES[strategy_name]

        base_dt = int(round(self.pm_downtime_base * self.level_downtime_mult[level]))
        base_dt = max(base_dt, 1)

        if self.spares[comp] > 0:
            self.spares[comp] -= 1
            new_dt = base_dt
            cost_mult = self.level_cost_mult[level]
            pm_cost += float(self.pm_costs[comp] * cost_mult)
            info["pm"] = f"{level}({strategy_name})_{self.comp_names[comp]}"
        else:
            new_dt = base_dt + self.lead_time + self.stockout_extra_downtime
            cost_mult = self.level_cost_mult[level]
            pm_cost += float(self.pm_costs[comp] * cost_mult)
            stockout_cost += float(self.stockout_extra_cost)
            info["pm"] = f"{level}_stockout({strategy_name})_{self.comp_names[comp]}"
            # Fix off-by-one: arrival time for stockout order matches auto-reorder
            if self.on_order_eta[comp] < 0:
                self.on_order_eta[comp] = self.t + 1 + self.lead_time
                self.on_order_qty[comp] = self.reorder_qty[comp]

        # Maintenance call count + extra penalty (added to PM cost)
        self.maint_times_used += 1
        if self.maint_extra_penalty > 0.0:
            pm_cost += float(self.maint_extra_penalty)

        self.down_remaining[comp] = max(self.down_remaining[comp], new_dt)

        hi_before = self.health[comp]
        gap = 1.0 - hi_before
        d_hi = max(strat["min_dHI"], strat["recover_ratio"] * gap)
        self.health[comp] = float(np.clip(hi_before + d_hi, 0.0, 1.2))

        self.degrade_rates[comp] *= float(strat["slope_factor"])
        self.noise_std[comp] = self.base_noise_std * float(strat["noise_mult"])

        return pm_cost, stockout_cost, info

    def _extract_hi_sequence(self, comp_idx, window=50):
        hist_arr = np.array(self.hi_history + [self.health.copy()], dtype=np.float32)
        seq = hist_arr[:, comp_idx]
        if len(seq) > window:
            seq = seq[-window:]
        return seq

    def _recommend_maintenance_from_hi(self, hi_seq, comp_idx):
        comp_name = self.comp_names[comp_idx]
        profile = self.business_profile
        hi_seq = np.asarray(hi_seq, dtype=np.float32)
        L = len(hi_seq)
        if L < 2:
            return f"Component {comp_name} HI sequence too short, recommend continued observation."

        hi_start = float(hi_seq[0])
        hi_now = float(hi_seq[-1])
        severity = 1.0 - hi_now

        short_L = min(5, L)
        short_seq = hi_seq[-short_L:]
        short_drop = float(short_seq[0] - short_seq[-1])
        volatility = (
            float(np.mean(np.abs(np.diff(hi_seq)))) if L > 2 else 0.0
        )

        if (hi_now <= self.failure_threshold + 0.05) or (
            severity >= 0.7
        ) or (short_drop > 0.15):
            level = "major"
        elif (hi_now <= self.request_threshold + 0.1) or (
            severity >= 0.4
        ) or (short_drop > 0.08):
            level = "medium"
        else:
            level = "minor"

        if profile == "aggressive":
            if (
                level == "major"
                and (hi_now > self.failure_threshold + 0.05)
                and (short_drop < 0.15)
            ):
                level = "medium"
            elif (
                level == "medium"
                and (hi_now > self.request_threshold)
                and (short_drop < 0.05)
            ):
                level = "minor"
        elif profile == "conservative":
            if level == "minor" and (severity > 0.35 or short_drop > 0.05):
                level = "medium"
            elif level == "medium" and (
                severity > 0.6 or hi_now < self.request_threshold
            ):
                level = "major"

        level_to_cn = {"minor": "Minor", "medium": "Medium", "major": "Major"}
        level_to_strategy = {
            "minor": "Minor Maintenance",
            "medium": "General Maintenance",
            "major": "Perfect Maintenance",
        }
        profile_desc = {
            "aggressive": "Aggressive (profit priority)",
            "balanced": "Balanced (profit & reliability)",
            "conservative": "Conservative (reliability priority)",
        }

        level_cn = level_to_cn[level]
        strat_name = level_to_strategy[level]
        prof_text = profile_desc.get(profile, profile)

        text = (
            f"[HI-driven Maintenance Recommendation] Component {comp_name}\n"
            f"- Current HI = {hi_now:.3f}, Start HI = {hi_start:.3f}, Recent {short_L}-step drop = {short_drop:.3f}, volatility={volatility:.3f}\n"
            f"- Thresholds: request={self.request_threshold:.2f}, failure={self.failure_threshold:.2f}\n"
            f"- Business profile: {prof_text}\n"
            f"=> Recommend {level_cn} maintenance (strategy {strat_name})."
        )
        return text

    def step(self, action):
        """
        Single step environment advance:
          1) Process maintenance action (may be limited by max_maint_times)
          2) Production side (buffer -> downstream)
          3) Logistics side: shipping/demand/inventory/backlog, compute profit
          4) HI degradation & request flags
          5) Output HI text recommendation on new trigger
          6) Failure & CM & downtime recovery
          7) Spare arrival & reorder
          8) Raw channel evolution (driven by process model, forming "world model")
          9) Compute reward decomposition (POSITIVE for good performance)
        """
        import random

        assert 0 <= action < self.n_actions
        info: Dict[str, Any] = {}
        profit = 0.0
        pm_cost = 0.0
        fail_cost = 0.0
        stockout_cost = 0.0

        old_request_flags = self.request_flags.copy()

        # 1) Maintenance action
        comp, level = self._decode_action(action)
        if comp is not None:
            pm_cost, stockout_cost, info = self._apply_maintenance_effect(
                comp, level, pm_cost, stockout_cost, info
            )

        # 2) Production segment: Stripper -> Separator/Compressor
        avail = (self.down_remaining == 0) & (~self.failed)
        strip_avail, sep_avail, comp_avail = map(bool, avail)

        strip_factor = self.health[0] if strip_avail else 0.0
        sep_factor = self.health[1] if sep_avail else 0.0
        comp_factor = self.health[2] if comp_avail else 0.0

        upstream_flow = self.capacity_per_step * strip_factor
        downstream_capacity = self.capacity_per_step * min(sep_factor, comp_factor)
        materials_available = self.buffer_inventory + upstream_flow
        production = min(materials_available, downstream_capacity)
        production = max(production, 0.0)
        self.buffer_inventory = materials_available - production
        self.buffer_inventory = float(
            np.clip(self.buffer_inventory, 0.0, self.buffer_capacity)
        )

        # 3) Logistics: FG inventory, demand, shipping, backlog
        self.fg_inventory += production
        demand = max(0.0, self.rng.normal(self.demand_mean, self.demand_std))
        total_demand = demand + self.backlog
        ship_qty = min(total_demand, self.fg_inventory, self.ship_capacity)
        self.fg_inventory -= ship_qty
        self.backlog = total_demand - ship_qty

        revenue = ship_qty * self.price_per_unit
        var_cost = production * self.variable_cost_per_unit
        holding_cost = self.fg_inventory * self.holding_cost_per_unit
        backlog_cost = self.backlog * self.backlog_penalty_per_unit
        ship_cost = ship_qty * self.ship_cost_per_unit
        spare_holding = float(self.spares.sum()) * self.spare_holding_cost_per_unit

        # Profit calculation: revenue - costs (should be positive when running well)
        profit = revenue - var_cost - holding_cost - backlog_cost - ship_cost - spare_holding

        # 4) HI degradation
        for i in range(self.n_components):
            if self.down_remaining[i] == 0 and not self.failed[i]:
                noise = self.rng.normal(0.0, self.noise_std[i])
                self.health[i] -= self.degrade_rates[i] + noise
                self.health[i] = float(np.clip(self.health[i], 0.0, 1.2))
                self.age[i] += 1.0

        # Request flags
        new_request_flags = self.health <= self.request_threshold
        newly_triggered = new_request_flags & (~old_request_flags)
        self.request_flags = new_request_flags

        # 5) HI text recommendation on new trigger
        hi_text_list = []
        for i in range(self.n_components):
            if newly_triggered[i]:
                hi_seq = self._extract_hi_sequence(i, window=50)
                text = self._recommend_maintenance_from_hi(hi_seq, i)
                hi_text_list.append(text)
        if hi_text_list:
            info["hi_recommendation"] = "\n\n".join(hi_text_list)

        # 6) Failure & CM
        for i in range(self.n_components):
            if self.health[i] <= self.failure_threshold and not self.failed[i]:
                self.failed[i] = True
                fail_cost += float(self.cm_costs[i])
                self.down_remaining[i] = max(self.down_remaining[i], self.cm_downtime)
                info[f"fail_{self.comp_names[i]}"] = True

        # Downtime recovery
        for i in range(self.n_components):
            if self.down_remaining[i] > 0:
                self.down_remaining[i] -= 1
                if self.down_remaining[i] == 0 and self.failed[i]:
                    self.failed[i] = False
                    self.health[i] = self.hi_reset
                    self.degrade_rates[i] = self.base_degrade_rates[i]
                    self.noise_std[i] = self.base_noise_std

        # 7) Spare arrival & reorder
        for i in range(self.n_components):
            if self.on_order_eta[i] >= 0 and self.t >= self.on_order_eta[i]:
                self.spares[i] = min(
                    int(self.max_spares), self.spares[i] + self.on_order_qty[i]
                )
                self.on_order_eta[i] = -1
                self.on_order_qty[i] = 0
            if self.spares[i] <= self.reorder_point[i] and self.on_order_eta[i] < 0:
                self.on_order_eta[i] = self.t + 1 + self.lead_time
                self.on_order_qty[i] = self.reorder_qty[i]

        # 8) Raw channel evolution (world model)
        if self.process_model is not None:
            self.raw_x = self.process_model.predict_next(self.raw_x, self.health)
        else:
            self.raw_x += self.rng.normal(0.0, 0.01, size=self.raw_x.shape).astype(
                np.float32
            )

        self.hi_history.append(self.health.copy())
        self.t += 1

        # 9) Reward decomposition - designed for POSITIVE returns
        w = self.vf_weights
        hi_penalty = float(np.sum(np.maximum(0.0, self.request_threshold - self.health)))
        risk_penalty = float(np.sum(self.failed.astype(np.float32)))

        # Reward: maximize profit, minimize costs and penalties
        reward = (
            w["w_profit"] * profit
            - w["w_maint"] * pm_cost
            - w["w_fail"] * fail_cost
            - w["w_stockout"] * stockout_cost
            - w["w_hi"] * (hi_penalty + risk_penalty)
        )

        done = self.t >= self.max_steps

        # === NEW: Env 自己的 HI / RUL 暴露出来 ===
        env_rul = self._compute_rul().astype(np.float32)  # shape (3,)

        info.update({
            "profit": profit,
            "pm_cost": pm_cost,
            "fail_cost": fail_cost,
            "stockout_cost": stockout_cost,
            "hi_penalty": hi_penalty,
            "risk_penalty": risk_penalty,
            "production": production,
            "ship_qty": ship_qty,
            "demand": demand,
            "fg_inventory": self.fg_inventory,
            "backlog": self.backlog,
            # === NEW: Env 真实健康与 RUL（考虑了维护/CM/降速等的物理状态） ===
            "env_health": self.health.copy(),  # shape (3,)
            "env_rul": env_rul.copy(),         # shape (3,)
        })

        return self._get_obs(), reward, done, info


def initialize_env_from_hi(
    env: PlantMaintenanceEnvV3,
    hi_traj: np.ndarray,
    trigger_step: int,
    infer_window: int = 20,
) -> PlantMaintenanceEnvV3:
    """
    Initialize env state from HI trajectory at trigger_step
    """
    hi_traj = np.asarray(hi_traj, dtype=np.float32)
    T = hi_traj.shape[0]
    trigger_step = max(0, min(trigger_step, T - 1))

    env.reset()
    env.health = hi_traj[trigger_step].copy()
    env.t = trigger_step

    # Infer degradation rate from recent HI slope
    start_idx = max(0, trigger_step - infer_window)
    for i in range(env.n_components):
        hi_slice = hi_traj[start_idx: trigger_step + 1, i]
        if len(hi_slice) >= 2:
            slope = (hi_slice[0] - hi_slice[-1]) / max(1, len(hi_slice) - 1)
            slope = max(slope, 1e-6)
            env.degrade_rates[i] = float(slope)

    env.request_flags = env.health <= env.request_threshold
    env.hi_history = [hi_traj[k].copy() for k in range(trigger_step + 1)]
    return env


class MaintenancePlanner:
    """
    Use same PlantMaintenanceEnvV3 + KAN process model for finite horizon rollout,
    compare "single maintenance at trigger step vs different maintenance actions" discounted returns.
    (This is a local single-step decision evaluator, main policy learning done by DQN.)
    """

    def __init__(
        self,
        env_cfg,
        process_model,
        plan_horizon=60,
        discount=0.99,
        business_profile="balanced",
    ):
        env_cfg = copy.deepcopy(env_cfg)
        env_cfg["business_profile"] = business_profile
        self.env_cfg = env_cfg
        self.process_model = process_model
        self.plan_horizon = int(plan_horizon)
        self.discount = float(discount)
        self.business_profile = business_profile

    def _make_env(self, seed=0):
        return PlantMaintenanceEnvV3(
            config=self.env_cfg,
            process_model=self.process_model,
            seed=seed,
        )

    def _build_candidate_actions(self, env: PlantMaintenanceEnvV3, hi_now: np.ndarray):
        hi_now = np.asarray(hi_now, dtype=np.float32)
        triggered = np.where(hi_now <= env.request_threshold)[0]
        actions = [0]
        if len(triggered) == 0:
            weakest = int(np.argmin(hi_now))
            triggered = [weakest]
        for idx in triggered:
            base = 1 + idx * len(env.maint_levels)
            for level_idx in range(len(env.maint_levels)):
                actions.append(base + level_idx)
        actions = sorted(set(actions))
        return actions, triggered

    def _rollout_with_first_action(
        self,
        hi_traj,
        trigger_step,
        first_action,
        seed=0,
    ):
        env = self._make_env(seed=seed)
        env = initialize_env_from_hi(env, hi_traj, trigger_step, infer_window=20)
        env.rng = np.random.RandomState(seed)

        horizon = self.plan_horizon
        gamma = 1.0
        total_return = 0.0

        metrics = {
            "discounted_return": 0.0,
            "profit_sum": 0.0,
            "pm_cost_sum": 0.0,
            "fail_cost_sum": 0.0,
            "stockout_cost_sum": 0.0,
            "production_sum": 0.0,
            "fail_count": 0,
        }

        for h in range(horizon):
            a = first_action if h == 0 else 0
            obs, r, done, info = env.step(a)
            total_return += gamma * r
            gamma *= self.discount

            metrics["discounted_return"] = total_return
            metrics["profit_sum"] += info.get("profit", 0.0)
            metrics["pm_cost_sum"] += info.get("pm_cost", 0.0)
            metrics["fail_cost_sum"] += info.get("fail_cost", 0.0)
            metrics["stockout_cost_sum"] += info.get("stockout_cost", 0.0)
            metrics["production_sum"] += info.get("production", 0.0)
            for j, name in enumerate(env.comp_names):
                if info.get(f"fail_{name}", False):
                    metrics["fail_count"] += 1
            if done:
                break

        return metrics, env

    def evaluate_hi_trajectory(
        self,
        hi_traj: np.ndarray,
        trigger_step: int,
        seed: int = 0,
    ):
        hi_traj = np.asarray(hi_traj, dtype=np.float32)
        env0 = self._make_env(seed=seed)
        env0 = initialize_env_from_hi(env0, hi_traj, trigger_step, infer_window=20)
        hi_now = env0.health.copy()

        candidate_actions, triggered_indices = self._build_candidate_actions(
            env0, hi_now
        )

        all_infos = []
        best_info = None
        for a in candidate_actions:
            metrics, env_last = self._rollout_with_first_action(
                hi_traj, trigger_step, a, seed=seed
            )
            comp, level = env_last._decode_action(a)
            level_name = level if level is not None else "none"
            comp_name = env_last.comp_names[comp] if comp is not None else "none"
            info_row = {
                "action": int(a),
                "component": comp_name,
                "level": level_name,
                "discounted_return": metrics["discounted_return"],
                "profit_sum": metrics["profit_sum"],
                "pm_cost_sum": metrics["pm_cost_sum"],
                "fail_cost_sum": metrics["fail_cost_sum"],
                "stockout_cost_sum": metrics["stockout_cost_sum"],
                "production_sum": metrics["production_sum"],
                "fail_count": metrics["fail_count"],
            }
            all_infos.append(info_row)
            if (best_info is None) or (
                metrics["discounted_return"] > best_info["discounted_return"]
            ):
                best_info = info_row

        text_lines = []
        text_lines.append("==== RL-style Maintenance Decision Evaluation Report (Single Step) ====")
        text_lines.append(f"Trigger step: t = {trigger_step}")
        text_lines.append(f"Current HI: {hi_now}")
        text_lines.append(f"Business profile: {self.business_profile}")
        text_lines.append("Candidate actions:")
        for row in all_infos:
            text_lines.append(
                f"  - action={row['action']:2d}, component={row['component']}, level={row['level']}, "
                f"discounted_return={row['discounted_return']:.2f}, "
                f"profit={row['profit_sum']:.2f}, PM_cost={row['pm_cost_sum']:.2f}, "
                f"fail_cost={row['fail_cost_sum']:.2f}, stockout_cost={row['stockout_cost_sum']:.2f}, "
                f"total_production={row['production_sum']:.2f}, fail_count={row['fail_count']}"
            )
        text_lines.append("\n>>> Recommended action (based on max discounted return):")
        ba = best_info
        text_lines.append(
            f"  * action={ba['action']:2d}, component={ba['component']}, level={ba['level']}, "
            f"discounted_return={ba['discounted_return']:.2f}, "
            f"profit={ba['profit_sum']:.2f}, PM_cost={ba['pm_cost_sum']:.2f}, "
            f"fail_cost={ba['fail_cost_sum']:.2f}, stockout_cost={ba['stockout_cost_sum']:.2f}, "
            f"total_production={ba['production_sum']:.2f}, fail_count={ba['fail_count']}"
        )

        if len(triggered_indices) == 1:
            idx = int(triggered_indices[0])
            hi_seq = hi_traj[max(0, trigger_step - 50): trigger_step + 1, idx]
            hi_text = env0._recommend_maintenance_from_hi(hi_seq, idx)
            text_lines.append("\n>>> HI shape heuristic recommendation (for comparison):")
            text_lines.append(hi_text)

        text_report = "\n".join(text_lines)
        return best_info, all_infos, text_report


class KANScenarioGenerator:
    def __init__(
        self,
        env_cfg,
        process_model,
        business_profile="balanced",
    ):
        env_cfg = copy.deepcopy(env_cfg)
        env_cfg["business_profile"] = business_profile
        self.env_cfg = env_cfg
        self.process_model = process_model
        self.business_profile = business_profile

    def _make_env(self, seed=0):
        return PlantMaintenanceEnvV3(
            config=self.env_cfg,
            process_model=self.process_model,
            seed=seed,
        )

    def generate_future(
        self,
        hi_traj: np.ndarray,
        trigger_step: int,
        first_action: int,
        horizon: int = 60,
        seed: int = 0,
    ):
        """
        Scenario 1: Only execute first_action at t=trigger_step, then all no-op.
        Returns:
          - raw_seq: (H+1, C) future H+1 steps raw sensor channels (from KAN process model + Env)
          - infos  : List[dict] info at each step
        Note: Future HI should be obtained by feeding raw_seq back to vaeSys_global_best.
        """
        hi_traj = np.asarray(hi_traj, dtype=np.float32)
        env = self._make_env(seed=seed)
        env = initialize_env_from_hi(env, hi_traj, trigger_step, infer_window=20)
        env.rng = np.random.RandomState(seed)

        raw_seq = [env.raw_x.copy()]
        infos = []

        for h in range(horizon):
            a = first_action if h == 0 else 0
            obs, r, done, info = env.step(a)
            raw_seq.append(env.raw_x.copy())
            infos.append(info)
            if done:
                break

        raw_seq = np.stack(raw_seq, axis=0)
        return raw_seq, infos

    def generate_future_with_policy(
        self,
        hi_traj: np.ndarray,
        trigger_step: int,
        agent,
        epsilon: float = 0.0,
        horizon: int = 60,
        seed: int = 0,
    ):
        """
        Scenario 2: From trigger_step, use DQN policy for each step's maintenance decision,
        to demonstrate "multiple maintenance timing" RL policy effect.

        Returns:
          - raw_seq: (H+1, C) future H+1 steps raw sensor channels
          - infos  : List[dict] info at each step

        Note: Future HI/RUL should be obtained by feeding raw_seq back to vaeSys_global_best.
        """
        hi_traj = np.asarray(hi_traj, dtype=np.float32)
        env = self._make_env(seed=seed)
        env = initialize_env_from_hi(env, hi_traj, trigger_step, infer_window=20)
        env.rng = np.random.RandomState(seed)

        raw_seq = [env.raw_x.copy()]
        infos = []

        for h in range(horizon):
            obs = env._get_obs()
            a = agent.act(obs, epsilon=epsilon)
            obs, r, done, info = env.step(a)
            raw_seq.append(env.raw_x.copy())
            infos.append(info)
            if done:
                break

        raw_seq = np.stack(raw_seq, axis=0)
        return raw_seq, infos

    @staticmethod
    def run_vaeSys_global_best(
        vaeSys_global_best,
        raw_seq: np.ndarray,
        env_id: int = 2,
    ):
        """
        Feed generated raw time series raw_seq back to SystemModel, get sysHi & predRUL
        (All HI/RUL produced by vaeSys_global_best)
        """
        if vaeSys_global_best is None:
            raise ValueError("vaeSys_global_best cannot be None")
        if not hasattr(vaeSys_global_best, "config"):
            raise ValueError("vaeSys_global_best missing config attribute")

        cfg = vaeSys_global_best.config
        seq_len = int(cfg["seqLength"])

        if "allComp" in cfg:
            first_comp = cfg["allComp"][0]
            in_dim_cfg_key = "in" + first_comp
            in_dim = int(cfg[in_dim_cfg_key])
        else:
            in_dim = raw_seq.shape[1]

        raw_seq = np.asarray(raw_seq, dtype=np.float32)
        if raw_seq.ndim != 2:
            raise ValueError(f"raw_seq must be (T, C), current shape = {raw_seq.shape}")
        T, C = raw_seq.shape

        if C > in_dim:
            raw_seq = raw_seq[:, :in_dim]
        elif C < in_dim:
            pad_feat = np.zeros((T, in_dim - C), dtype=np.float32)
            raw_seq = np.concatenate([raw_seq, pad_feat], axis=1)

        if T >= seq_len:
            seq = raw_seq[-seq_len:, :]
        else:
            pad_len = seq_len - T
            pad = np.repeat(raw_seq[0:1, :], pad_len, axis=0)
            seq = np.concatenate([pad, raw_seq], axis=0)

        seqData = tf.convert_to_tensor(seq[None, :, :], dtype=tf.float32)
        seqCycle = tf.zeros((1, 1), dtype=tf.float32)
        env_tf = tf.constant(env_id, dtype=tf.int32)

        zCr1, zCr2, zCv2, sysHi, predictedRul, hiMatrixOutput = (
            vaeSys_global_best.getHIandRUL((seqData, seqCycle, env_tf))
        )

        return {
            "zCr1": zCr1.numpy(),
            "zCr2": zCr2.numpy(),
            "zCv2": zCv2.numpy(),
            "sysHi": sysHi.numpy(),
            "predRUL": predictedRul.numpy(),
            "hiMatrix": hiMatrixOutput.numpy(),
        }


# ============================================================
# C. DQN + Replay Buffer (TensorFlow version, with best model save/load)
# ============================================================

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done", "episode_id")
)


class DuelingDQN(keras.Model):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.feature = keras.Sequential([
            layers.Dense(hidden_dim, activation="relu"),
            layers.Dense(hidden_dim, activation="relu"),
        ])
        self.value_stream = keras.Sequential([
            layers.Dense(hidden_dim, activation="relu"),
            layers.Dense(1),
        ])
        self.adv_stream = keras.Sequential([
            layers.Dense(hidden_dim, activation="relu"),
            layers.Dense(action_dim),
        ])

    def call(self, x):
        x = self.feature(x)
        v = self.value_stream(x)
        a = self.adv_stream(x)
        q = v + a - tf.reduce_mean(a, axis=1, keepdims=True)
        return q

    def get_config(self):
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class RhymeReplayBuffer:
    """
    Replay Buffer with PER + "similar HI draft priority sampling"
    Enhanced with high-risk priority boosting for failure/stockout events.
    Now includes HI cache for vectorized similarity computation.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        n_channels: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        eps: float = 1e-5,
        high_risk_priority_boost: float = 2.0,
    ):
        self.capacity = int(capacity)
        self.buffer: list[Transition] = []
        self.priorities: list[float] = []
        self.episode_ids: list[int] = []
        self.high_risk_flags: list[bool] = []  # Track high-risk transitions
        self.hi_cache: list[np.ndarray] = []  # Cache HI slices for vectorized sampling
        self.pos = 0

        self.obs_dim = int(obs_dim)
        self.n_channels = int(n_channels)

        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_frames = int(beta_frames)
        self.eps = float(eps)
        self.max_priority = 1.0
        self.high_risk_priority_boost = float(high_risk_priority_boost)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
        episode_id: int,
        is_high_risk: bool = False,
    ):
        # Apply priority boost for high-risk transitions (failure/stockout)
        priority = self.max_priority
        if is_high_risk:
            priority *= self.high_risk_priority_boost

        # Cache HI slice for vectorized similarity computation
        hi_slice = self._get_hi_slice(state)

        if len(self.buffer) < self.capacity:
            self.buffer.append(
                Transition(state, action, reward, next_state, done, episode_id)
            )
            self.priorities.append(priority)
            self.episode_ids.append(episode_id)
            self.high_risk_flags.append(is_high_risk)
            self.hi_cache.append(hi_slice)
        else:
            self.buffer[self.pos] = Transition(
                state, action, reward, next_state, done, episode_id
            )
            self.priorities[self.pos] = priority
            self.episode_ids[self.pos] = episode_id
            self.high_risk_flags[self.pos] = is_high_risk
            self.hi_cache[self.pos] = hi_slice
        self.pos = (self.pos + 1) % self.capacity

    def _beta_by_step(self, step: int) -> float:
        return min(
            1.0,
            self.beta_start
            + step * (1.0 - self.beta_start) / max(1, self.beta_frames),
        )

    def _get_hi_slice(self, state: np.ndarray) -> np.ndarray:
        """
        Extract HI segment from obs for "similar HI draft priority sampling"
        obs = [raw_norm (n_channels), hi(3), age(3), rul(3),
               spares(3), req(3), t, fg, backlog, maint_budget]
        """
        s = np.asarray(state, dtype=np.float32)
        hi_start = self.n_channels
        hi_end = self.n_channels + 3
        return s[hi_start:hi_end]

    def _compute_probs(self, indices: np.ndarray) -> np.ndarray:
        if len(self.buffer) == self.capacity:
            prios = np.asarray(self.priorities, dtype=np.float32)
        else:
            prios = np.asarray(self.priorities[: len(self.buffer)], dtype=np.float32)
        p = prios[indices] ** self.alpha
        p_sum = p.sum()
        if p_sum <= 0:
            p = np.ones_like(p) / len(p)
        else:
            p /= p_sum
        return p

    def sample(self, batch_size: int, global_step: int):
        N = len(self.buffer)
        if N == 0:
            raise ValueError("Replay buffer is empty.")
        batch_size = min(batch_size, N)

        indices = np.random.choice(N, batch_size, replace=False)
        probs = self._compute_probs(indices)
        beta = self._beta_by_step(global_step)

        weights = (N * probs) ** (-beta)
        weights /= weights.max() + 1e-8
        weights = np.asarray(weights, dtype=np.float32)

        batch = [self.buffer[i] for i in indices]
        batch = Transition(*zip(*batch))
        return batch, indices, weights

    def sample_with_draft(
        self,
        batch_size: int,
        global_step: int,
        ref_state: np.ndarray,
        current_episode: int,
        epoch_window: int = 15,
        draft_ratio: float = 0.4,
    ):
        """
        Part of samples from "recent episodes + similar HI" segments with priority,
        rest sampled normally by PER.
        Uses vectorized HI similarity computation via hi_cache.
        """
        N = len(self.buffer)
        if N == 0:
            raise ValueError("Replay buffer is empty.")
        batch_size = min(batch_size, N)

        draft_n = int(batch_size * draft_ratio)
        draft_n = max(1, min(draft_n, batch_size))

        recent_min = max(0, current_episode - epoch_window)
        candidate_indices = [
            i for i, eid in enumerate(self.episode_ids) if eid >= recent_min
        ]
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)

        if len(candidate_indices) < draft_n:
            return self.sample(batch_size, global_step)

        # Vectorized HI similarity computation using cached HI slices
        ref_hi = self._get_hi_slice(ref_state).reshape(1, -1)  # (1, 3)
        candidate_hi = np.stack(
            [self.hi_cache[i] for i in candidate_indices],
            axis=0,
        )  # (M, 3)
        dist = np.linalg.norm(candidate_hi - ref_hi, axis=1)  # (M,)
        sims = 1.0 / (dist + 1e-3)

        per_probs = self._compute_probs(candidate_indices)
        p = per_probs * sims
        p_sum = p.sum()
        if p_sum <= 0:
            p = np.ones_like(p) / len(p)
        else:
            p /= p_sum

        draft_indices = np.random.choice(
            candidate_indices, size=draft_n, replace=False, p=p
        )

        remain_n = batch_size - draft_n
        if remain_n > 0:
            remain_batch, remain_idx, remain_w = self.sample(
                remain_n, global_step
            )
            remain_indices = np.asarray(remain_idx, dtype=np.int64)
        else:
            remain_indices = np.asarray([], dtype=np.int64)

        all_indices = np.concatenate([draft_indices, remain_indices], axis=0)

        N = len(self.buffer)
        probs_all = self._compute_probs(all_indices)
        beta = self._beta_by_step(global_step)

        weights = (N * probs_all) ** (-beta)
        weights /= weights.max() + 1e-8
        weights = np.asarray(weights, dtype=np.float32)

        batch = [self.buffer[i] for i in all_indices]
        batch = Transition(*zip(*batch))
        return batch, all_indices, weights

    def update_priorities(self, indices, td_errors: np.ndarray):
        td_errors = np.abs(td_errors) + self.eps
        for idx, err in zip(indices, td_errors):
            # Maintain high-risk boost when updating priorities
            boost = self.high_risk_priority_boost if self.high_risk_flags[idx] else 1.0
            self.priorities[idx] = float(err) * boost
            self.max_priority = max(self.max_priority, float(err))


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 5e-4,
        gamma: float = 0.99,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = float(gamma)

        self.q_net = DuelingDQN(obs_dim, action_dim, hidden_dim)
        self.target_net = DuelingDQN(obs_dim, action_dim, hidden_dim)

        # Build networks
        dummy_input = tf.zeros((1, obs_dim))
        self.q_net(dummy_input)
        self.target_net(dummy_input)
        self.target_net.set_weights(self.q_net.get_weights())

        self.optimizer = keras.optimizers.Adam(learning_rate=lr)

        # Compile the update function for performance
        self._compiled_update = tf.function(self._update_inner, autograph=False)

    def act(self, state: np.ndarray, epsilon: float) -> int:
        """
        epsilon-greedy action; use epsilon=0.0 for eval.
        """
        import random

        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        s = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        q = self.q_net(s)
        return int(tf.argmax(q, axis=1).numpy()[0])

    def _update_inner(
        self,
        states_tf,
        actions_tf,
        rewards_tf,
        next_states_tf,
        dones_tf,
        weights_tf,
    ):
        """
        Inner update function compiled with tf.function for performance.
        """
        with tf.GradientTape() as tape:
            # Double DQN target
            next_q_online = self.q_net(next_states_tf)
            next_actions = tf.argmax(next_q_online, axis=1)
            next_q_target = self.target_net(next_states_tf)
            next_q_values = tf.gather(next_q_target, next_actions, batch_dims=1)
            targets = rewards_tf + self.gamma * (1.0 - dones_tf) * next_q_values

            q_values = self.q_net(states_tf)
            actions_onehot = tf.one_hot(actions_tf, self.action_dim)
            q_selected = tf.reduce_sum(q_values * actions_onehot, axis=1)
            td_errors = targets - q_selected
            loss = tf.reduce_mean(weights_tf * tf.square(td_errors))

        grads = tape.gradient(loss, self.q_net.trainable_variables)
        grads = [tf.clip_by_norm(g, 10.0) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))

        return td_errors, loss

    def update(self, batch: Transition, weights: np.ndarray):
        """
        Update Q-network using a batch of transitions.
        Uses compiled tf.function for performance.
        """
        states = np.stack(batch.state).astype(np.float32)
        actions = np.array(batch.action, dtype=np.int32)
        rewards = np.array(batch.reward, dtype=np.float32)
        next_states = np.stack(batch.next_state).astype(np.float32)
        dones = np.array(batch.done, dtype=np.float32)

        states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tf = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards_tf = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states_tf = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones_tf = tf.convert_to_tensor(dones, dtype=tf.float32)
        weights_tf = tf.convert_to_tensor(weights, dtype=tf.float32)

        td_errors_tf, loss_tf = self._compiled_update(
            states_tf, actions_tf, rewards_tf,
            next_states_tf, dones_tf, weights_tf,
        )

        return td_errors_tf.numpy(), float(loss_tf.numpy())

    def update_target(self):
        self.target_net.set_weights(self.q_net.get_weights())

    def save(self, path: str):
        """
        Save current policy and target network weights (TensorFlow format).
        """
        self.q_net.save_weights(path + "_q_net.weights.h5")
        self.target_net.save_weights(path + "_target_net.weights.h5")

    def load(self, path: str):
        """
        Load policy and target network weights from saved files.
        """
        self.q_net.load_weights(path + "_q_net.weights.h5")
        self.target_net.load_weights(path + "_target_net.weights.h5")



# ==============================================================
# ==============================================================
# ==============================================================
# ==============================================================
# This class is mainly used for the maintenance activity
# ==============================================================
# ==============================================================
# ==============================================================
# ==============================================================
class MaintenanceActivity():
    def __init__(self):


        self.clientConstruction = ClientConstruction()
        self.preBuiltAi         = PreBuildAI()

    def linear_epsilon(self,    step: int,
                                eps_start: float,
                                eps_end: float,
                                decay_steps: int,) -> float:
        frac = max(0.0, (decay_steps - step) / max(1, decay_steps))
        return eps_end + (eps_start - eps_end) * frac


    def train_dqn_with_rhymerl(self,
        env,
        agent: DQNAgent,
        num_episodes: int = 50,
        max_steps_per_episode: int | None = None,
        buffer_capacity: int = 100_000,
        batch_size: int = 256,
        min_buffer_size: int = 1_000,
        train_updates_per_step: int = 2,
        update_interval: int = 4,
        eps_start: float = 1.0,
        eps_end: float = 0.15,
        eps_decay_steps: int = 30_000,
        target_update_interval: int = 1_000,
        seed: int = 0,
        log_step_rewards: bool = False,
        save_best_path: str | None = None,
        print_episode_summary_every: int = 1,
        # Exploration enhancement parameters
        no_improve_eps_reset_threshold: int = 5,
        no_improve_eps_reset_value: float = 0.3,
        exploration_episode_interval: int = 5,
        exploration_episode_eps: float = 0.3,
        draft_disable_steps: int = 10_000,
        high_risk_eps_boost: float = 0.15,
        # New: log episode actions
        log_episode_actions: bool = False,
    ):
        """
        Core training loop with enhanced exploration:
        - Epsilon reset when no improvement for N episodes
        - Periodic high-exploration episodes
        - Disable draft sampling for first K steps
        - High-risk region targeted exploration (failure/stockout)
        - Batch training every N steps (update_interval) for efficiency
        """
        import random

        tf.random.set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        if max_steps_per_episode is None:
            max_steps_per_episode = env.max_steps

        obs0 = env.reset()
        obs_dim = obs0.shape[0]
        replay = RhymeReplayBuffer(
            capacity=buffer_capacity,
            obs_dim=obs_dim,
            n_channels=env.n_channels,
            alpha=0.6,
            beta_start=0.4,
            beta_frames=num_episodes * max_steps_per_episode,
            high_risk_priority_boost=2.0,
        )

        global_step = 0
        episode_returns: list[float] = []

        best_return = -float("inf")
        best_ep = -1
        episodes_since_improvement = 0
        current_eps_override = None  # For epsilon reset mechanism

        for ep in range(num_episodes):
            state = env.reset()
            ep_return = 0.0

            profit_sum = 0.0
            pm_cost_sum = 0.0
            fail_cost_sum = 0.0
            stockout_cost_sum = 0.0
            hi_penalty_sum = 0.0
            risk_penalty_sum = 0.0

            last_eps = 0.0

            # Record actions for this episode
            ep_actions: list[int] = []

            # Determine if this is a high-exploration episode
            is_exploration_episode = (ep % exploration_episode_interval == 0)

            for t in range(max_steps_per_episode):
                # Base epsilon from linear schedule
                base_epsilon = self.linear_epsilon(
                    global_step, eps_start, eps_end, eps_decay_steps
                )

                # Apply epsilon override if set (from no-improvement reset)
                if current_eps_override is not None:
                    epsilon = max(base_epsilon, current_eps_override)
                else:
                    epsilon = base_epsilon

                # Apply exploration episode boost
                if is_exploration_episode:
                    epsilon = max(epsilon, exploration_episode_eps)

                last_eps = epsilon

                action = agent.act(state, epsilon)
                next_state, reward, done, info = env.step(action)

                # Record action for this episode
                ep_actions.append(int(action))

                # Detect high-risk events for priority boosting and exploration
                is_high_risk = (
                    info.get("fail_cost", 0.0) > 0.0 or
                    info.get("stockout_cost", 0.0) > 0.0
                )

                # Temporarily boost epsilon after high-risk event for next step
                if is_high_risk:
                    epsilon = min(1.0, epsilon + high_risk_eps_boost)

                if log_step_rewards:
                    print(
                        f"[step] g={global_step:6d}, ep={ep:3d}, t={t:3d}, a={action:2d}"
                    )

                replay.push(
                    state, action, reward, next_state, float(done),
                    episode_id=ep, is_high_risk=is_high_risk
                )

                state = next_state
                ep_return += reward
                global_step += 1

                profit_sum += info["profit"]
                pm_cost_sum += info["pm_cost"]
                fail_cost_sum += info["fail_cost"]
                stockout_cost_sum += info["stockout_cost"]
                hi_penalty_sum += info["hi_penalty"]
                risk_penalty_sum += info["risk_penalty"]

                # Batch training every update_interval steps
                if len(replay) >= min_buffer_size and (global_step % update_interval == 0):
                    total_updates = train_updates_per_step * update_interval
                    for _ in range(total_updates):
                        # Disable draft sampling for first K steps
                        if global_step < draft_disable_steps:
                            batch, indices, weights = replay.sample(
                                batch_size, global_step
                            )
                        else:
                            batch, indices, weights = replay.sample_with_draft(
                                batch_size=batch_size,
                                global_step=global_step,
                                ref_state=state,
                                current_episode=ep,
                                epoch_window=15,
                                draft_ratio=0.4,
                            )
                        td_errors, loss = agent.update(batch, weights)
                        replay.update_priorities(indices, td_errors)

                if global_step % target_update_interval == 0:
                    agent.update_target()

                if done:
                    break

            episode_returns.append(ep_return)

            w = env.vf_weights
            reward_profit = w["w_profit"] * profit_sum
            reward_maint = -w["w_maint"] * pm_cost_sum
            reward_fail = -w["w_fail"] * fail_cost_sum
            reward_stockout = -w["w_stockout"] * stockout_cost_sum
            reward_hi = -w["w_hi"] * (hi_penalty_sum + risk_penalty_sum)
            reward_sum_check = (
                reward_profit
                + reward_maint
                + reward_fail
                + reward_stockout
                + reward_hi
            )

            if (ep + 1) % max(1, print_episode_summary_every) == 0:
                print(
                    f"[DQN Train] Ep {ep+1:3d}/{num_episodes}, "
                    f"return={ep_return: .3e}, "
                    f"profit={profit_sum: .2f}, pm={pm_cost_sum: .2f}, "
                    f"fail={fail_cost_sum: .2f}, stockout={stockout_cost_sum: .2f}, "
                    f"hi_penalty={hi_penalty_sum: .3f}, risk_penalty={risk_penalty_sum: .3f}, "
                    f"eps={last_eps:.3f}, reward_decomp={reward_sum_check: .2f}, "
                    f"maint_times_used={env.maint_times_used}, "
                    f"explore_ep={is_exploration_episode}"
                )

                # Print episode actions if enabled
                if log_episode_actions:
                    max_show = 50
                    actions_to_show = ep_actions[:max_show]
                    print(
                        f"[DQN Train] Ep {ep+1:3d} actions "
                        f"(first {len(actions_to_show)}/{len(ep_actions)} steps):"
                    )
                    print("   ", actions_to_show)

            # Record and save best policy
            if ep_return > best_return:
                best_return = ep_return
                best_ep = ep
                episodes_since_improvement = 0
                current_eps_override = None  # Clear epsilon override on improvement
                if save_best_path is not None:
                    agent.save(save_best_path)
                    print(
                        f"[DQN Train] >>> New best model at ep={ep+1}, "
                        f"return={ep_return:.2f}, saved to {save_best_path}"
                    )
            else:
                episodes_since_improvement += 1

            # Epsilon reset mechanism when no improvement for N episodes
            if episodes_since_improvement >= no_improve_eps_reset_threshold:
                current_eps_override = no_improve_eps_reset_value
                print(
                    f"[DQN Train] No improvement for {episodes_since_improvement} episodes, "
                    f"resetting epsilon to {no_improve_eps_reset_value:.2f}"
                )
                episodes_since_improvement = 0  # Reset counter

        return {
            "returns": episode_returns,
            "global_steps": global_step,
            "best_return": best_return,
            "best_episode": best_ep,
        }


    def train_dqn_agent_for_maintenance(self,
        env_cfg: Dict[str, Any],
        process_model: NextStepKANProcessModel,
        num_episodes: int = 50,
        seed: int = 42,
        save_best_path: Optional[str] = None,
        log_step_rewards: bool = False,
        log_episode_actions: bool = False,
    ) -> Tuple[DQNAgent, Dict[str, Any]]:
        """
        One-time training of maintenance DQN:
        - Uses given env_cfg + process_model to construct training environment
        - Returns trained agent and training statistics
        - If save_best_path is not None, will auto-save "best performing policy" parameters.
        """
        import random

        env_train = PlantMaintenanceEnvV3(
            config=copy.deepcopy(env_cfg),
            process_model=process_model,
            seed=seed,
        )
        obs_dim = env_train.reset().shape[0]
        action_dim = env_train.n_actions

        print(
            f"[INFO] DQN obs_dim = {obs_dim}, action_dim = {action_dim}, "
            f"n_channels = {env_train.n_channels}"
        )

        agent = DQNAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=128,
            lr=5e-4,
            gamma=0.99,
        )

        train_stats = self.train_dqn_with_rhymerl(
            env=env_train,
            agent=agent,
            num_episodes=num_episodes,
            max_steps_per_episode=env_train.max_steps,
            buffer_capacity=100_000,
            batch_size=256,
            min_buffer_size=1_000,
            train_updates_per_step=2,
            update_interval=4,
            eps_start=1.0,
            eps_end=0.15,
            eps_decay_steps=30_000,
            target_update_interval=1_000,
            seed=seed,
            log_step_rewards=log_step_rewards,
            save_best_path=save_best_path,
            print_episode_summary_every=1,
            # Enhanced exploration parameters
            no_improve_eps_reset_threshold=5,
            no_improve_eps_reset_value=0.3,
            exploration_episode_interval=5,
            exploration_episode_eps=0.3,
            draft_disable_steps=10_000,
            high_risk_eps_boost=0.15,
            log_episode_actions=log_episode_actions,
        )

        return agent, train_stats


    def create_process_model(self,n_channels: int, n_components: int = 3, hidden_dim: int = 64) -> NextStepKANProcessModel:
        """
        Create and build a NextStepKANProcessModel instance.
        """
        model = NextStepKANProcessModel(
            n_channels=n_channels,
            n_components=n_components,
            hidden_dim=hidden_dim,
        )
        # Build the model with dummy input
        dummy_input = tf.zeros((1, n_channels + n_components))
        model(dummy_input)
        return model


    def save_process_model(self,model: NextStepKANProcessModel, path: str):
        """
        Save process model weights to disk.
        """
        model.save_weights(path + ".weights.h5")
        # Also save config as JSON
        config = model.get_config()
        import json
        with open(path + "_config.json", "w") as f:
            json.dump(config, f)


    def load_process_model(self,path: str) -> NextStepKANProcessModel:
        """
        Load process model from saved weights and config.
        """
        import json
        with open(path + "_config.json", "r") as f:
            config = json.load(f)
        model = NextStepKANProcessModel(**config)
        # Build the model with dummy input
        dummy_input = tf.zeros((1, config["n_channels"] + config["n_components"]))
        model(dummy_input)
        model.load_weights(path + ".weights.h5")
        return model


    def create_dqn_agent(self,obs_dim: int, action_dim: int, hidden_dim: int = 128) -> DQNAgent:
        """
        Create a DQNAgent instance with specified dimensions.
        """
        return DQNAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr=5e-4,
            gamma=0.99,
        )


    def load_dqn_agent_for_maintenance(self,
        env_cfg: Dict[str, Any],
        process_model: NextStepKANProcessModel,
        checkpoint_path: str,
    ) -> DQNAgent:
        """
        Deployment/inference usage:
        - Given env_cfg and process_model consistent with training
        - Load trained best policy parameters from checkpoint_path
        - Return DQNAgent ready for act(...)
        """
        env_eval = PlantMaintenanceEnvV3(
            config=copy.deepcopy(env_cfg),
            process_model=process_model,
            seed=0,
        )
        obs_dim = env_eval.reset().shape[0]
        action_dim = env_eval.n_actions

        agent = DQNAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=128,
            lr=5e-4,
            gamma=0.99,
        )
        agent.load(checkpoint_path)
        return agent


    # ============================================================
    # D. Plotting utility functions (not auto-called in run_rl_maintenance_scenario)
    # ============================================================


    def plot_hi_history_and_future(self,
        hi_traj: "np.ndarray",
        hi_future_maint: "np.ndarray",
        hi_future_none: "np.ndarray",
        trigger_step: int,
        hi_future_maint_corr: "np.ndarray | None" = None,
        hi_future_none_corr: "np.ndarray | None" = None,
        title: str = "HI history & future (maint vs no maint)",
    ):
        """
        Plot HI history and future trajectories.
        Optionally includes corrected HI trajectories from equivalent age correction.

        Note: hi_future_maint / hi_future_none should come from SystemModel(vaeSys_global_best),
        not from Env/KAN internal health variable.
        """
        T_hist = hi_traj.shape[0]
        x_hist = np.arange(T_hist)
        Hm = hi_future_maint.shape[0]
        Hn = hi_future_none.shape[0]
        x_future_m = np.arange(trigger_step, trigger_step + Hm)
        x_future_n = np.arange(trigger_step, trigger_step + Hn)
        comp_names = ["Stripper", "Separator", "Compressor"]

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
        for i in range(3):
            ax = axes[i]
            ax.plot(x_hist, hi_traj[:, i], label="History")
            ax.plot(
                x_future_m,
                hi_future_maint[:, i],
                linestyle="--",
                label="Future (DQN policy)" if i == 0 else None,
            )
            ax.plot(
                x_future_n,
                hi_future_none[:, i],
                linestyle=":",
                label="Future (no maint)" if i == 0 else None,
            )
            # Plot corrected HI if provided
            if hi_future_maint_corr is not None:
                ax.plot(
                    x_future_m,
                    hi_future_maint_corr[:, i],
                    linestyle="-.",
                    color="green",
                    label="Future (DQN corrected)" if i == 0 else None,
                )
            if hi_future_none_corr is not None:
                ax.plot(
                    x_future_n,
                    hi_future_none_corr[:, i],
                    linestyle="-.",
                    color="orange",
                    label="Future (no maint corrected)" if i == 0 else None,
                )
            ax.axvline(trigger_step, linestyle="--", color="k")
            ax.set_ylabel(f"{comp_names[i]} HI")
            ax.grid(True, linestyle=":")
        axes[-1].set_xlabel("Time step")
        axes[0].legend(loc="best")
        fig.suptitle(title)
        fig.tight_layout()
        plt.show()


    def plot_raw_channels_compare(self,
        raw_future_maint: "np.ndarray",
        raw_future_none: "np.ndarray",
        trigger_step: int,
        title: str = "Raw channels compare (maint vs no maint)",
    ):
        Hm, C = raw_future_maint.shape
        Hn, C2 = raw_future_none.shape
        assert C == C2
        x_m = np.arange(trigger_step, trigger_step + Hm)
        x_n = np.arange(trigger_step, trigger_step + Hn)

        n_cols = 5
        n_rows = int(math.ceil(C / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(3 * n_cols, 2 * n_rows), sharex=True
        )
        axes = np.atleast_2d(axes)

        for ch in range(C):
            r = ch // n_cols
            c = ch % n_cols
            ax = axes[r, c]
            ax.plot(x_m, raw_future_maint[:, ch], label="DQN policy")
            ax.plot(x_n, raw_future_none[:, ch], alpha=0.7, label="no maint")
            ax.set_title(f"ch {ch}")
            ax.grid(True, linestyle=":")

        for ch in range(C, n_rows * n_cols):
            r = ch // n_cols
            c = ch % n_cols
            axes[r, c].axis("off")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")
        fig.suptitle(title)
        fig.tight_layout()
        plt.show()


    # ============================================================
    # E. Unified entry point: RL + scenario simulation from HI repository
    # ============================================================


    def _extract_tail_env_metrics(self, rollout_infos: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the last step's Env HI / Env RUL from a scenario's infos list.
        Assumes rollout_infos is non-empty and each info contains 'env_health' / 'env_rul'.
        """
        if not rollout_infos:
            raise ValueError("rollout_infos is empty, cannot extract tail Env metrics.")
        last = rollout_infos[-1]
        env_health_tail = np.asarray(last["env_health"], dtype=np.float32)  # (3,)
        env_rul_tail = np.asarray(last["env_rul"], dtype=np.float32)        # (3,)
        return env_health_tail, env_rul_tail


    def compute_equivalent_age_factors(self,
        env_health_policy: np.ndarray,
        env_rul_policy: np.ndarray,
        env_health_nomaint: np.ndarray,
        env_rul_nomaint: np.ndarray,
        failure_threshold_env: float = 0.2,
        clip_min: float = 0.5,
        clip_max: float = 2.0,
    ) -> Tuple[float, float]:
        """
        Compute RUL and HI margin equivalent age ratios.

        Args:
            env_health_policy: Env HI at end of policy scenario (3,)
            env_rul_policy: Env RUL at end of policy scenario (3,)
            env_health_nomaint: Env HI at end of no-maintenance scenario (3,)
            env_rul_nomaint: Env RUL at end of no-maintenance scenario (3,)
            failure_threshold_env: Failure threshold for HI margin calculation
            clip_min: Minimum clipping value for ratios
            clip_max: Maximum clipping value for ratios

        Returns:
            phi_R: RUL equivalent age ratio
            phi_H: HI margin equivalent age ratio
        """
        eps = 1e-6
        env_health_policy = np.asarray(env_health_policy, dtype=np.float32)
        env_rul_policy = np.asarray(env_rul_policy, dtype=np.float32)
        env_health_nomaint = np.asarray(env_health_nomaint, dtype=np.float32)
        env_rul_nomaint = np.asarray(env_rul_nomaint, dtype=np.float32)

        # ---- RUL equivalent age ratio ----
        r_pol_mean = float(np.mean(env_rul_policy))
        r_nom_mean = float(np.mean(env_rul_nomaint)) + eps
        phi_R = r_pol_mean / r_nom_mean

        # ---- HI margin equivalent age ratio ----
        margin_pol = np.maximum(env_health_policy - failure_threshold_env, 0.0)
        margin_nom = np.maximum(env_health_nomaint - failure_threshold_env, 0.0)
        m_pol_mean = float(np.mean(margin_pol))
        m_nom_mean = float(np.mean(margin_nom)) + eps
        phi_H = m_pol_mean / m_nom_mean

        phi_R = float(np.clip(phi_R, clip_min, clip_max))
        phi_H = float(np.clip(phi_H, clip_min, clip_max))

        return phi_R, phi_H


    def apply_equivalent_age_correction_hi_rul(self,
        sys_eval_policy: Dict[str, Any],
        sys_eval_nomaint: Dict[str, Any],
        phi_R: float,
        phi_H: float,
        failure_threshold_sys: float = 0.2,
        hi_clip_min: float = 0.0,
        hi_clip_max: float = 1.2,
    ) -> Dict[str, Any]:
        """
        Apply equivalent age correction to SystemModel's RUL and HI predictions.

        Args:
            sys_eval_policy: SystemModel output for policy scenario
            sys_eval_nomaint: SystemModel output for no-maintenance scenario
            phi_R: RUL equivalent age ratio
            phi_H: HI margin equivalent age ratio
            failure_threshold_sys: Failure threshold for HI margin calculation
            hi_clip_min: Minimum clipping value for corrected HI
            hi_clip_max: Maximum clipping value for corrected HI

        Returns:
            Dictionary containing raw and corrected RUL/HI values
        """
        # ---------- RUL ----------
        R_sys_policy_raw = float(np.asarray(sys_eval_policy["predRUL"]).reshape(-1)[-1])
        R_sys_nomaint_raw = float(np.asarray(sys_eval_nomaint["predRUL"]).reshape(-1)[-1])
        R_corr_policy = max(0.0, phi_R * R_sys_policy_raw)
        R_corr_nomaint = R_sys_nomaint_raw

        # ---------- System HI ----------
        sysHi_policy_raw = float(np.asarray(sys_eval_policy["sysHi"]).reshape(-1)[-1])
        sysHi_nomaint_raw = float(np.asarray(sys_eval_nomaint["sysHi"]).reshape(-1)[-1])
        h_sys_policy = 0.5 * (np.tanh(sysHi_policy_raw) + 1.0)
        h_sys_nomaint = 0.5 * (np.tanh(sysHi_nomaint_raw) + 1.0)

        margin_sys_policy = max(h_sys_policy - failure_threshold_sys, 0.0)
        margin_sys_corr_policy = phi_H * margin_sys_policy
        h_sys_corr_policy = np.clip(
            failure_threshold_sys + margin_sys_corr_policy,
            hi_clip_min,
            hi_clip_max,
        )
        h_sys_corr_nomaint = np.clip(h_sys_nomaint, hi_clip_min, hi_clip_max)

        # ---------- Component HI (hiMatrix last timestep) ----------
        hi_comp_policy_raw = None
        hi_comp_nomaint_raw = None
        hi_comp_corr_policy = None
        hi_comp_corr_nomaint = None

        if "hiMatrix" in sys_eval_policy and "hiMatrix" in sys_eval_nomaint:
            hiMat_pol = np.asarray(sys_eval_policy["hiMatrix"], dtype=np.float32)
            hiMat_nom = np.asarray(sys_eval_nomaint["hiMatrix"], dtype=np.float32)

            def _extract_last_comp_hi(mat: np.ndarray, name: str) -> np.ndarray:
                if mat.ndim == 3:

                    return mat[0, -1, :]
                elif mat.ndim == 2:
                    return mat[-1, :]
                elif mat.ndim == 1:
                    return mat
                else:
                    raise ValueError(f"Unexpected hiMatrix shape for {name}: {mat.shape}")

            hi_comp_policy_raw = _extract_last_comp_hi(hiMat_pol, "policy")
            hi_comp_nomaint_raw = _extract_last_comp_hi(hiMat_nom, "nomaint")

            margin_comp_policy = np.maximum(hi_comp_policy_raw - failure_threshold_sys, 0.0)
            margin_comp_corr_policy = phi_H * margin_comp_policy
            hi_comp_corr_policy = np.clip(
                failure_threshold_sys + margin_comp_corr_policy,
                hi_clip_min,
                hi_clip_max,
            )
            hi_comp_corr_nomaint = np.clip(
                hi_comp_nomaint_raw,
                hi_clip_min,
                hi_clip_max,
            )

        return {
            "phi_R": float(phi_R),
            "phi_H": float(phi_H),
            "R_sys_policy_raw": R_sys_policy_raw,
            "R_sys_nomaint_raw": R_sys_nomaint_raw,
            "R_corr_policy": R_corr_policy,
            "R_corr_nomaint": R_corr_nomaint,
            "HI_sys_policy_raw": float(h_sys_policy),
            "HI_sys_nomaint_raw": float(h_sys_nomaint),
            "HI_sys_corr_policy": float(h_sys_corr_policy),
            "HI_sys_corr_nomaint": float(h_sys_corr_nomaint),
            "HI_comp_policy_raw": hi_comp_policy_raw,
            "HI_comp_nomaint_raw": hi_comp_nomaint_raw,
            "HI_comp_corr_policy": hi_comp_corr_policy,
            "HI_comp_corr_nomaint": hi_comp_corr_nomaint,
        }


    def run_rl_maintenance_scenario(self,
        hi_repo: Dict[int, Dict[str, Dict[str, List["np.ndarray"]]]],
        env_cfg: Dict[str, Any],
        process_model: NextStepKANProcessModel,
        agent: DQNAgent,
        vaeSys_global_best,
        env_id: int,
        split_name: str,
        traj_idx: int,
        trigger_step: Optional[int] = None,
        business_profile: str = "balanced",
        plan_horizon: int = 60,
    ) -> Dict[str, Any]:
        """
        Using pre-computed hi_repo, on specified env/split/traj:
            1) Auto or manual select trigger_step
            2) Use extended PlantMaintenanceEnvV3 + existing DQN policy, run multi-step maintenance scenario
            3) Use KANScenarioGenerator to compare:
                - DQN policy scenario (multiple maintenance)
                - Never maintain scenario
            4) All history & future HI / RUL are officially from SystemModel(vaeSys_global_best);
            KAN process model only generates future raw sensor channels raw_seq.
            5) Apply equivalent age correction to SystemModel's HI/RUL predictions

        Returns only the "official" HI/RUL from vaeSys_global_best (not internal env HI),
        plus equivalent age corrected values.
        """
        assert env_id in hi_repo, f"env_id={env_id} not in hi_repo"
        assert split_name in hi_repo[env_id], f"split_name={split_name} not in hi_repo[{env_id}]"

        hi_list = hi_repo[env_id][split_name]["hi"]
        sys_hi_list = hi_repo[env_id][split_name]["sys_hi"]

        num_traj = len(hi_list)
        if traj_idx < 0 or traj_idx >= num_traj:
            raise ValueError(
                f"traj_idx={traj_idx} out of range, current env={env_id}, split={split_name} has {num_traj} trajectories."
            )

        hi_traj = hi_list[traj_idx]
        sys_hi_traj = sys_hi_list[traj_idx]
        T_hist = hi_traj.shape[0]
        print(
            f"[INFO] Using env={env_id}, split='{split_name}' trajectory {traj_idx}, "
            f"HI sequence length T = {T_hist}"
        )

        env_cfg_local = copy.deepcopy(env_cfg)
        env_cfg_local["business_profile"] = business_profile

        req_thr = float(env_cfg_local.get("request_threshold", 0.4))
        fail_thr = float(env_cfg_local.get("failure_threshold", 0.2))

        if trigger_step is None or trigger_step < 0 or trigger_step >= T_hist:
            trigger_step_final = self.clientConstruction.auto_select_trigger_step(
                hi_traj=hi_traj,
                sys_hi_traj=sys_hi_traj,
                request_threshold=req_thr,
                failure_threshold=fail_thr,
            )
            print(f"[INFO] Auto-selected trigger_step = {trigger_step_final}")
        else:
            trigger_step_final = int(trigger_step)
            print(f"[INFO] Using externally specified trigger_step = {trigger_step_final}")

        # At this trigger step, use DQN to see first action
        env_eval = PlantMaintenanceEnvV3(
            config=env_cfg_local,
            process_model=process_model,
            seed=999,
        )
        env_eval = initialize_env_from_hi(
            env_eval,
            hi_traj=hi_traj,
            trigger_step=trigger_step_final,
            infer_window=20,
        )
        obs_eval = env_eval._get_obs()
        dqn_action_at_trigger = agent.act(obs_eval, epsilon=0.0)
        comp_idx, level_name = env_eval._decode_action(dqn_action_at_trigger)
        comp_name = env_eval.comp_names[comp_idx] if comp_idx is not None else "none"

        print(
            f"\n[DQN] First maintenance action at trigger_step={trigger_step_final}: "
            f"action={dqn_action_at_trigger}, component={comp_name}, level={level_name}"
        )

        # Planner local single-step evaluation
        planner = MaintenancePlanner(
            env_cfg=env_cfg_local,
            process_model=process_model,
            plan_horizon=plan_horizon,
            discount=0.98,
            business_profile=business_profile,
        )
        best_info, all_infos, text_report = planner.evaluate_hi_trajectory(
            hi_traj=hi_traj,
            trigger_step=trigger_step_final,
            seed=42,
        )
        print("\n" + "=" * 80)
        print("[MaintenancePlanner Single-Step Maintenance Evaluation Report]")
        print(text_report)

        metrics_dqn_local, _ = planner._rollout_with_first_action(
            hi_traj=hi_traj,
            trigger_step=trigger_step_final,
            first_action=dqn_action_at_trigger,
            seed=1234,
        )
        print("\n>>> DQN first action local evaluation based on this HI trajectory:")
        print(
            f"  action={dqn_action_at_trigger}, component={comp_name}, level={level_name}, "
            f"discounted_return={metrics_dqn_local['discounted_return']:.2f}, "
            f"profit={metrics_dqn_local['profit_sum']:.2f}, PM_cost={metrics_dqn_local['pm_cost_sum']:.2f}, "
            f"fail_cost={metrics_dqn_local['fail_cost_sum']:.2f}, stockout_cost={metrics_dqn_local['stockout_cost_sum']:.2f}, "
            f"total_production={metrics_dqn_local['production_sum']:.2f}, fail_count={metrics_dqn_local['fail_count']}"
        )

        generator = KANScenarioGenerator(
            env_cfg=env_cfg_local,
            process_model=process_model,
            business_profile=business_profile,
        )
        horizon = plan_horizon

        # DQN policy scenario: future raw sequence (HI evaluated by SystemModel afterwards)
        raw_future_dqn, rollout_infos_dqn = generator.generate_future_with_policy(
            hi_traj=hi_traj,
            trigger_step=trigger_step_final,
            agent=agent,
            epsilon=0.0,
            horizon=horizon,
            seed=2024,
        )

        # Never maintain scenario: future raw sequence
        raw_future_none, rollout_infos_none = generator.generate_future(
            hi_traj=hi_traj,
            trigger_step=trigger_step_final,
            first_action=0,
            horizon=horizon,
            seed=2024,
        )

        print("\n[INFO] Future raw shapes:", raw_future_dqn.shape, raw_future_none.shape)

        # SystemModel evaluates future HI & RUL for both scenarios (all from vaeSys_global_best)
        sys_eval_dqn = generator.run_vaeSys_global_best(
            vaeSys_global_best=vaeSys_global_best,
            raw_seq=raw_future_dqn,
            env_id=env_id,
        )
        print("\n=== SystemModel Output (DQN policy scenario) ===")
        print("sysHi:", sys_eval_dqn["sysHi"])
        print("predRUL:", sys_eval_dqn["predRUL"])

        sys_eval_none = generator.run_vaeSys_global_best(
            vaeSys_global_best=vaeSys_global_best,
            raw_seq=raw_future_none,
            env_id=env_id,
        )
        print("\n=== SystemModel Output (Never maintain scenario) ===")
        print("sysHi:", sys_eval_none["sysHi"])
        print("predRUL:", sys_eval_none["predRUL"])

        # === NEW: Apply equivalent age correction based on Env HI/RUL ===
        # 1) Extract tail Env metrics from rollout infos
        env_health_dqn, env_rul_dqn = self._extract_tail_env_metrics(rollout_infos_dqn)
        env_health_none, env_rul_none = self._extract_tail_env_metrics(rollout_infos_none)

        # 2) Compute equivalent age ratios (using Env's failure_threshold)
        failure_thr_env = float(env_cfg_local.get("failure_threshold", 0.2))
        phi_R, phi_H = self.compute_equivalent_age_factors(
            env_health_policy=env_health_dqn,
            env_rul_policy=env_rul_dqn,
            env_health_nomaint=env_health_none,
            env_rul_nomaint=env_rul_none,
            failure_threshold_env=failure_thr_env,
            clip_min=0.5,
            clip_max=2.0,
        )

        # 3) Apply these ratios to correct SystemModel's RUL + HI
        hi_rul_equiv = self.apply_equivalent_age_correction_hi_rul(
            sys_eval_policy=sys_eval_dqn,
            sys_eval_nomaint=sys_eval_none,
            phi_R=phi_R,
            phi_H=phi_H,
            failure_threshold_sys=failure_thr_env,  # Use Env threshold; can be configured separately if needed
            hi_clip_min=0.0,
            hi_clip_max=1.2,
        )

        print("\n=== Equivalent Age Correction Results ===")
        print(f"phi_R (RUL ratio): {hi_rul_equiv['phi_R']:.3f}")
        print(f"phi_H (HI margin ratio): {hi_rul_equiv['phi_H']:.3f}")
        print(f"RUL policy (raw -> corrected): {hi_rul_equiv['R_sys_policy_raw']:.2f} -> {hi_rul_equiv['R_corr_policy']:.2f}")
        print(f"HI sys policy (raw -> corrected): {hi_rul_equiv['HI_sys_policy_raw']:.3f} -> {hi_rul_equiv['HI_sys_corr_policy']:.3f}")

        # Return both raw and corrected HI/RUL from vaeSys_global_best
        result = {
            "env_id": env_id,
            "split_name": split_name,
            "traj_idx": traj_idx,
            "trigger_step": trigger_step_final,
            "env_cfg": env_cfg_local,
            "raw_future_dqn": raw_future_dqn,
            "raw_future_none": raw_future_none,
            "rollout_infos_dqn": rollout_infos_dqn,
            "rollout_infos_none": rollout_infos_none,
            "dqn_action_at_trigger": dqn_action_at_trigger,
            "dqn_component_at_trigger": comp_name,
            "dqn_level_at_trigger": level_name,
            "dqn_metrics_local": metrics_dqn_local,
            "planner_best_info": best_info,
            "planner_all_infos": all_infos,
            "planner_report": text_report,
            # SystemModel raw predictions
            "system_model_future_eval_dqn": sys_eval_dqn,
            "system_model_future_eval_none": sys_eval_none,
            # NEW: Equivalent age corrected RUL + HI
            "equivalent_age_hi_rul": hi_rul_equiv,
        }
        return result


    def run_rl_maintenance_scenario_with_checkpoint(self,
        hi_repo: Dict[int, Dict[str, Dict[str, List["np.ndarray"]]]],
        env_cfg: Dict[str, Any],
        process_model: NextStepKANProcessModel,
        vaeSys_global_best,
        checkpoint_path: str,
        env_id: int,
        split_name: str,
        traj_idx: int,
        trigger_step: Optional[int] = None,
        business_profile: str = "balanced",
        plan_horizon: int = 60,
    ) -> Dict[str, Any]:
        """
        Convenience entry:
        - Load trained best DQN policy from disk (checkpoint_path)
        - Run multi-step maintenance + world model simulation on given env/split/traj (from hi_repo)
        - Return same structure result dict as run_rl_maintenance_scenario
        """
        agent = self.load_dqn_agent_for_maintenance(
            env_cfg=env_cfg,
            process_model=process_model,
            checkpoint_path=checkpoint_path,
        )
        return self.run_rl_maintenance_scenario(
            hi_repo=hi_repo,
            env_cfg=env_cfg,
            process_model=process_model,
            agent=agent,
            vaeSys_global_best=vaeSys_global_best,
            env_id=env_id,
            split_name=split_name,
            traj_idx=traj_idx,
            trigger_step=trigger_step,
            business_profile=business_profile,
            plan_horizon=plan_horizon,
        )
    

    # ----------------------------------------
    # This function is run before running the last
    # one
    # ----------------------------------------
    def preRunMaintenancePrediction(self,):
            # ============================================================
        # 0. Prerequisites:
        #    - preData, norData, originalData, vaeSys_global_best
        #      are assumed to be constructed and imported externally
        #      (consistent with your original script setup)
        # ============================================================

        # Step 1: Pre-compute HI repository (A1 + A2 + A3)
        print(">>> Step 1: Pre-compute HI repository (A1+A2+A3)...")
        hi_data = self.clientConstruction.precompute_hi_repository(
            preData             =   self.preBuiltAi.preData,
            norData             =   self.preBuiltAi.norData,
            originalData        =   self.preBuiltAi.originalData,
            vaeSys_global_best  =   self.preBuiltAi.vaeSys_global_best,
        )
        hi_repo = hi_data["hi_repo"]

        # ------------------------------------------------------------
        # 1.1 Function to retrieve already-split train / val / test data
        #     (does not re-split; directly uses results from precompute_hi_repository)
        # ------------------------------------------------------------
        def get_clients_by_env_from_hi_data(hi_data_dict):
            """
            Retrieve the already-split train/val/test data from precompute_hi_repository.

            Input:
                hi_data_dict: Return dictionary from precompute_hi_repository:
                    {
                    "clients_by_env": {2:{...},3:{...},4:{...}},
                    "eval_results":   {...},
                    "hi_repo":        {...}
                    }

            Returns:
                clients_by_env = {
                    2: {"train":..., "val":..., "test":...},
                    3: {...},
                    4: {...},
                }
            """
            if "clients_by_env" not in hi_data_dict:
                raise KeyError(
                    "hi_data is missing 'clients_by_env'. Please verify the return value of precompute_hi_repository."
                )
            return hi_data_dict["clients_by_env"]

        clients_by_env = get_clients_by_env_from_hi_data(hi_data)
        print(">>> clients_by_env loaded; available for subsequent use where original train/val/test is needed")

        # ============================================================
        # 2. Deployment configuration + paths + hi_repo persistence (fix FileNotFoundError)
        # ============================================================


        # ========= 2.1 Paths & Basic Configuration =========
        deployment_config = {
            # Base directory
            "save_dir": "./fl_and_maintenance/pretrained_model_and_agents",
            # Model paths (keep consistent with __main__ / training)
            "process_model_path": "process_model_best.keras",
            "dqn_agent_path": "dqn_maint_best.keras",
            "hi_repo_path": "hi_repo_env234.pkl",  # Path where hi_repo is saved

            # Select the scenario to visualize: environment, dataset split, and trajectory index
            "env_id": 2,              # 2 / 3 / 4
            "split_name": "test",     # "train" / "val" / "test"
            "traj_idx": 10,           # Trajectory index
            "trigger_step": None,     # None means auto-select trigger point; can also be an integer

            # Business settings
            "business_profile": "balanced",
            "plan_horizon": 60,
        }

        # Derived full paths
        SAVE_DIR = deployment_config["save_dir"]
        PROCESS_MODEL_PATH = os.path.join(SAVE_DIR, deployment_config["process_model_path"])
        DQN_AGENT_PATH = os.path.join(SAVE_DIR, deployment_config["dqn_agent_path"])
        HI_REPO_PATH = os.path.join(SAVE_DIR, deployment_config["hi_repo_path"])

        # Convenience variables for scenario selection
        ENV_ID = deployment_config["env_id"]
        SPLIT_NAME = deployment_config["split_name"]
        TRAJ_IDX = deployment_config["traj_idx"]
        TRIGGER_STEP = deployment_config["trigger_step"]
        BUSINESS_PROFILE = deployment_config["business_profile"]
        PLAN_HORIZON = deployment_config["plan_horizon"]

        # ========= 2.2 Handle hi_repo save / load to avoid FileNotFoundError =========
        os.makedirs(SAVE_DIR, exist_ok=True)

        if os.path.exists(HI_REPO_PATH):
            # If file already exists, load it (can be reused across sessions)
            print(f"[INFO] Found existing HI_REPO_PATH = {HI_REPO_PATH}, loading...")
            with open(HI_REPO_PATH, "rb") as f:
                loaded_obj = pickle.load(f)
            if isinstance(loaded_obj, dict) and "hi_repo" in loaded_obj:
                hi_repo = loaded_obj["hi_repo"]
            else:
                hi_repo = loaded_obj
            print("[INFO] hi_repo loaded from pickle file")
        else:
            # If file does not exist (the FileNotFoundError case you encountered),
            # use the hi_data / hi_repo just computed, and save it for future direct loading.
            print(
                f"[WARN] HI_REPO_PATH '{HI_REPO_PATH}' not found, "
                "using hi_repo from precompute_hi_repository and saving it."
            )
            with open(HI_REPO_PATH, "wb") as f:
                # For future flexibility, save the entire hi_data (includes clients_by_env, eval_results, hi_repo)
                pickle.dump(hi_data, f)
            print(f"[INFO] hi_data (including hi_repo) saved to {HI_REPO_PATH}")

        # ============================================================
        # 3. Construct env_cfg consistent with training
        # ============================================================
        env_cfg = {
            "n_channels": len(self.preBuiltAi.preData.headline["allWithoutLabelNoBias"]),
            "failure_threshold": 0.2,
            "request_threshold": 0.4,
            # Production & logistics parameters (keep consistent with training)
            "capacity_per_step": 100.0,
            "buffer_capacity": 1000.0,
            "buffer_init": 500.0,
            "price_per_unit": 15.0,
            "variable_cost_per_unit": 3.0,
            "demand_mean": 80.0,
            "demand_std": 12.0,
            "ship_capacity": 120.0,
            "holding_cost_per_unit": 0.1,
            "backlog_penalty_per_unit": 0.5,
            "ship_cost_per_unit": 0.05,
            "fg_init": 200.0,
            # Spare parts & logistics
            "initial_spares": [5, 5, 5],
            "reorder_point": [2, 2, 2],
            "reorder_qty": [3, 3, 3],
            "lead_time": 7,
            "stockout_extra_downtime": 3,
            "stockout_extra_cost": 200.0,
            "spare_holding_cost_per_unit": 0.2,
            # Maintenance count limit & extra cost
            "max_maint_times": 10,
            "maint_extra_penalty": 10.0,
            # Episode length
            "max_steps": 365,
            # Default business profile
            "business_profile": BUSINESS_PROFILE,
        }


        # ============================================================
        # 4. Load pre-trained KAN process model + DQN maintenance policy
        # ============================================================

        # 4.1 Load pre-trained KAN process model
        process_model = self.load_process_model(PROCESS_MODEL_PATH)

        # 4.2 Load pre-trained DQN agent (maintenance policy)
        agent = self.load_dqn_agent_for_maintenance(
            env_cfg=env_cfg,
            process_model=process_model,
            checkpoint_path=DQN_AGENT_PATH,
        )
        return hi_repo, env_cfg, process_model, agent
    


    # -------------------------------------
    # This function is used to run the 
    # maintenance function
    # -------------------------------------
    def runMaintenancePrediction(self,  hi_repo             = None, 
                                        env_cfg             = None, 
                                        process_model       = None, 
                                        agent               = None, 
                                        vaeSys_global_best  = None, 
                                        env_id              = 4, 
                                        splitData           = "train",
                                        trajectIdx          = 1,
                                        atSimulationTime    = 200,
                                        businessProfile     = "balanced",
                                        predictionSteps     = 60):
        


        
        result = self.run_rl_maintenance_scenario(  hi_repo             =   hi_repo,
                                                    env_cfg             =   env_cfg,
                                                    process_model       =   process_model,
                                                    agent               =   agent,
                                                    vaeSys_global_best  =   vaeSys_global_best,     # Assuming this is already loaded externally
                                                    env_id              =   env_id,                 # "ENV_ID":2,3,4,
                                                    split_name          =   splitData,              # "SPLIT_NAME":train, val,test
                                                    traj_idx            =   trajectIdx,             # "TRAJ_IDX":write as you want within the number of tragetory in train,val or test
                                                    trigger_step        =   atSimulationTime,       # "TRIGGER_STEP" :write as you want
                                                    business_profile    =   businessProfile,        # "BUSINESS_PROFILE":"aggressive" "balanced" "conservative"
                                                    plan_horizon        =   predictionSteps         # "PLAN_HORIZON"
                                                    )

        print("-------------------------------=================================")
        # print(result['planner_best_info'])
        print(f'The planplanner_best_info {result["planner_best_info"]}')
        print(f'The planner_all_infos {result["planner_all_infos"]}')
        print(f'The planner_report {result["planner_report"]}')

        print("\n===== Trigger Recommendation Results (Key Fields) =====")
        print("Trigger step:", result["trigger_step"])
        print("DQN recommended action: action =", result["dqn_action_at_trigger"])
        print("Maintenance target component:", result["dqn_component_at_trigger"])
        print("Maintenance level:", result["dqn_level_at_trigger"])

        print("\n[Local Single-step Planner Report (Text Version)]:")
        print(result["planner_report"])

        print("\n[System RUL / HI after Equivalent Age Correction]:")
        print(result["equivalent_age_hi_rul"])



        # # Historical HI trajectory (from hi_repo, evaluated earlier by SystemModel)
        # hi_hist = hi_repo[ENV_ID][SPLIT_NAME]["hi"][TRAJ_IDX]

        # # Future raw sequences (DQN policy vs never maintain)
        # raw_future_dqn  = result["raw_future_dqn"]   # (H+1, C)
        # raw_future_none = result["raw_future_none"]  # (H+1, C) No maintenance data
        return result


# if __name__ == "__main__":
#     maintenance = MaintenanceActivity()

#     hi_repo,        \
#     env_cfg,        \
#     process_model,  \
#     agent       = maintenance.preRunMaintenancePrediction()

#     mainResults = maintenance.runMaintenancePrediction( hi_repo             = hi_repo, 
#                                                         env_cfg             = env_cfg, 
#                                                         process_model       = process_model, 
#                                                         agent               = agent, 
#                                                         vaeSys_global_best  = maintenance.preBuiltAi.vaeSys_global_best, 
#                                                         env_id              = 4, 
#                                                         splitData           = "train",
#                                                         trajectIdx          = 1,
#                                                         atSimulationTime    = 200,
#                                                         businessProfile     = "balanced",
#                                                         predictionSteps     = 60)
