# =============================================
# IMPORT
# =============================================

from model.tepSystemModel   import SystemModel

import numpy                as np
import os


import shutil
import hashlib
import h5py
# from tensorflow.keras import backend as K



# =============================================================================
# Helpers
# =============================================================================
class FLAndMaintenance():
    def __init__(self,config = None): 

        TMP_MIRROR_ROOT = "./checkpoint/flCkpt/_tmp_variant_mirror/"
        os.makedirs(TMP_MIRROR_ROOT, exist_ok=True)
        component_dir_root = TMP_MIRROR_ROOT
        self.vaeSys_contributor = SystemModel(config=config, componentModelPath=component_dir_root)




    def _comp_dir(self, root: str, sub: str, use_hi: bool) -> str:
        """Return the directory that holds numbered variants for a component."""
        return os.path.join(root, f"{sub}{'_HI_updates' if use_hi else ''}")

    def _ensure_empty_dir(self, path: str):
        """Ensure a directory exists and is empty."""
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    def _md5(self, path: str) -> str:
        """MD5 of the entire file (debug)."""
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _first_float_stats(self, h5_path: str, max_items: int = 3):
        """
        Return up to 'max_items' (name, shape, mean, std) for float datasets in an HDF5 file.
        This gives you a quick fingerprint that should differ between original and perturbed variants.
        """
        out = []
        with h5py.File(h5_path, "r") as f:
            def visitor(n, o):
                if len(out) >= max_items:
                    return
                if isinstance(o, h5py.Dataset) and np.issubdtype(o.dtype, np.floating):
                    a = o[()]
                    out.append((n, a.shape, float(a.mean()), float(a.std())))
            f.visititems(visitor)
        return out

    def _print_h5_debug(self, label: str, path: str):
        """Print MD5 and a few dataset stats for quick verification."""
        print(f"[DEBUG] {label}: {path}")
        try:
            print(f"        MD5: { self._md5(path) }")
            stats = self._first_float_stats(path)
            if stats:
                for i, (n, shp, m, s) in enumerate(stats, 1):
                    print(f"        ds#{i}: name={n}, shape={shp}, mean={m:.6f}, std={s:.6f}")
            else:
                print("        (no float datasets found)")
        except Exception as e:
            print(f"        (debug read failed: {e})")

    def _copy_variant(self, src_dir: str, dst_dir: str, mapping_with_suffix: dict, idx: int, debug_compare_to: str = None):
        """
        Copy numbered variant files and rename them to expected standard names.
        Example mapping: {"encoderWts": "encoderWts"} will copy encoderWts_7.h5 -> encoderWts.h5
        """
        os.makedirs(dst_dir, exist_ok=True)
        for expected_name, base_stem in mapping_with_suffix.items():
            # Numbered variant (e.g., encoderWts_7.h5)
            src = os.path.join(src_dir, f"{base_stem}_{idx}.h5")
            dst = os.path.join(dst_dir, f"{expected_name}.h5")

            if not os.path.isfile(src):
                raise FileNotFoundError(f"Variant file not found: {src}")

            print(f"[COPY] {src}  ->  {dst}")
            shutil.copy2(src, dst)

            # Debug: compare to the original (if provided)
            if debug_compare_to is not None:
                orig = os.path.join(debug_compare_to, base_stem + ".h5")
                if os.path.isfile(orig):
                    self._print_h5_debug("ORIG ", orig)
                else:
                    print(f"[DEBUG] ORIG not found (skipped): {orig}")
            self._print_h5_debug("MIRROR", dst)  # show what we actually load afterwards


    # =============================================================================
    # Main function to load variant model
    # =============================================================================
    def load_variant_model(self, config, variant_idx=7, use_hi_updates=True):
        """
        Load a perturbed model variant with specified configuration.

        Args:
            config: Configuration dictionary containing model parameters
            variant_idx: Index of the variant to load (e.g., 7 for encoderWts_7.h5)
            use_hi_updates: Whether to use variants from "_HI_updates" folders

        Returns:
            vaeSys_contributor: Loaded SystemModel instance with variant weights
        """
        # =============================================================================
        # Variant selection
        # =============================================================================
        VARIANT_IDX = variant_idx
        USE_HI_UPDATES = use_hi_updates

        # Where your generated variants live (the destination from your perturbation script)
        VARIANTS_ROOT = "./checkpoint/flCkpt/HI_model_update/"

        # For sanity checks against originals (optional but recommended)
        ORIGINALS_ROOT = "./checkpoint/flCkpt/bestModel_5epoch/"

        # Temporary mirror: we will copy and rename numbered files here to the canonical names
        TMP_MIRROR_ROOT = "./checkpoint/flCkpt/_tmp_variant_mirror/"
        # os.makedirs(TMP_MIRROR_ROOT, exist_ok=True)

        # =============================================================================
        # Source & destination directories (4 components)
        # =============================================================================
        SRC_REAC_DIR = self._comp_dir(VARIANTS_ROOT, "bestModelReac", USE_HI_UPDATES)
        SRC_SEP_DIR = self._comp_dir(VARIANTS_ROOT, "bestModelSep", USE_HI_UPDATES)
        SRC_STRIP_DIR = self._comp_dir(VARIANTS_ROOT, "bestModelStrip", USE_HI_UPDATES)
        SRC_SYS_DIR = self._comp_dir(VARIANTS_ROOT, "bestModelSys", USE_HI_UPDATES)

        DST_REAC_DIR = os.path.join(TMP_MIRROR_ROOT, "bestModelReac")
        DST_SEP_DIR = os.path.join(TMP_MIRROR_ROOT, "bestModelSep")
        DST_STRIP_DIR = os.path.join(TMP_MIRROR_ROOT, "bestModelStrip")
        DST_SYS_DIR = os.path.join(TMP_MIRROR_ROOT, "bestModelSys")

        # Rebuild mirror directories fresh
        self._ensure_empty_dir(DST_REAC_DIR)
        self._ensure_empty_dir(DST_SEP_DIR)
        self._ensure_empty_dir(DST_STRIP_DIR)
        self._ensure_empty_dir(DST_SYS_DIR)

        # Each component requires three files; system head needs the HI+RUL file
        COMP_MAPPING = {
            "encoderWts": "encoderWts",
            "decoderWts": "decoderWts",
            "maskWts": "maskWts",
        }
        SYS_MAPPING = {
            "hiAndRulWts": "hiAndRulWts",
        }

        print("=== Copying selected variants into the mirror (with debug fingerprints) ===")
        self._copy_variant(SRC_REAC_DIR, DST_REAC_DIR, COMP_MAPPING, VARIANT_IDX,
                    debug_compare_to=os.path.join(ORIGINALS_ROOT, "bestModelReac"))
        self._copy_variant(SRC_SEP_DIR, DST_SEP_DIR, COMP_MAPPING, VARIANT_IDX,
                    debug_compare_to=os.path.join(ORIGINALS_ROOT, "bestModelSep"))
        self._copy_variant(SRC_STRIP_DIR, DST_STRIP_DIR, COMP_MAPPING, VARIANT_IDX,
                    debug_compare_to=os.path.join(ORIGINALS_ROOT, "bestModelStrip"))
        self._copy_variant(SRC_SYS_DIR, DST_SYS_DIR, SYS_MAPPING, VARIANT_IDX,
                    debug_compare_to=os.path.join(ORIGINALS_ROOT, "bestModelSys"))

        print("\n=== Mirror is ready. About to build and load the model from the mirror ===\n")

        required_keys = ["seqLength", "inReac"]
        missing = [k for k in required_keys if k not in config]
        if missing:
            raise KeyError(f"Missing required config keys: {missing}")

        # Force System mode and ensure default HI selection dimension (adjust if your code differs)
        config["component"] = "Sys"
        config.setdefault("hiSelectionDim", 120)

        # The model should read weights from these subfolders when `freezeComponents` is True
        component_dir_root = TMP_MIRROR_ROOT

        # Instantiate your model
        vaeSys_contributor = SystemModel(config=config, componentModelPath=component_dir_root)

        # Build the model to initialize layers. This shape list must match your architecture.
        vaeSys_contributor.build(input_shape=[
            (None, config["seqLength"], config["inReac"]),  # main sequence
            (None, 1, config["inReac"]),                    # initial state
            (None, 1),                                      # scalar/time feature
            (None, config["hiSelectionDim"]),               # HI selection vector
        ])

        # Print the summary (optional but helpful)
        vaeSys_contributor.summary()

        # Prepare the canonical directories we just mirrored to
        modelDirComp1 = os.path.join(component_dir_root, "bestModelReac")
        modelDirComp2 = os.path.join(component_dir_root, "bestModelSep")
        modelDirComp3 = os.path.join(component_dir_root, "bestModelStrip")
        modelDirSys = os.path.join(component_dir_root, "bestModelSys")

        print("\n=== Loading weights from the MIRROR ===")
        print("Reac:  ", modelDirComp1)
        print("Sep:   ", modelDirComp2)
        print("Strip: ", modelDirComp3)
        print("Sys:   ", modelDirSys)
        vaeSys_contributor.load_weights(modelDirComp1, modelDirComp2, modelDirComp3, modelDirSys)
        print("✓ Weights loaded from the selected variant ("
            f"idx={VARIANT_IDX}, HI_updates={USE_HI_UPDATES}).")

        return vaeSys_contributor
    
    def load_global_best(self,):
    
        curDir = "./checkpoint/flCkpt/bestModel_200epoch/"
        modelDirComp1 = curDir + "bestModelReac"
        modelDirComp2 = curDir + "bestModelSep"
        modelDirComp3 = curDir + "bestModelStrip"
        modelDirSys   = curDir + "bestModelSys"

        self.vaeSys_contributor.load_weights(modelDirComp1, modelDirComp2, modelDirComp3, modelDirSys)
        print("Loaded the best weight!!!!!!!!!!!!!")

        return self.vaeSys_contributor

# =============================================================================
# Example usage / function call
# =============================================================================
# if __name__ == "__main__":

#     # Load the variant model
#     # You can change variant_idx (0-9) and use_hi_updates (True/False) as needed
#     vaeSys_contributor = load_variant_model(
#         config=config,
#         variant_idx=1,# We pertubte 10 different model,just choose any of them you want to see
#         use_hi_updates=True
#     )

#     print("\n✅ Model loaded successfully and ready for use!")