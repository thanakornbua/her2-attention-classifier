"""
Macenko stain normalization with optional CuPy GPU acceleration for numeric operations.
The GPU backend is only used for numerical computation; inputs/outputs remain NumPy arrays.
"""

import numpy as np

# SciPy used only for CPU eigen decomposition when CuPy isn't used
from scipy.linalg import eigh as cpu_eigh


class MacenkoNormalizer:
    def __init__(self, alpha=0.15, beta=0.10, percentiles=(1, 99), use_gpu=False, cupy_module=None):
        """
        :param alpha: angle param (kept for API compatibility, not used directly)
        :param beta: angle param (kept for API compatibility, not used directly)
        :param percentiles: tuple (low, high) percentiles used when calculating max concentrations
        :param use_gpu: if True, attempt to use CuPy for array computations
        :param cupy_module: optional explicit cupy module (for testing); if None and use_gpu True, tries to import cupy
        """
        self.alpha = alpha
        self.beta = beta
        self.percentiles = percentiles
        self.use_gpu = bool(use_gpu)

        # Lazily set xp (numpy or cupy) — keep inputs/outputs as NumPy
        if self.use_gpu:
            if cupy_module is not None:
                self.cp = cupy_module
            else:
                try:
                    import cupy as cp
                    self.cp = cp
                except Exception as e:
                    raise ImportError("CuPy requested but not available: {}".format(e))
            self.xp = self.cp
        else:
            self.cp = None
            self.xp = np

    def _to_backend(self, arr):
        """
        Convert a NumPy array to backend array (CuPy) if GPU is enabled, otherwise return the NumPy array.
        """
        if self.use_gpu:
            return self.cp.asarray(arr)
        return arr

    def _to_numpy(self, arr):
        """
        Convert backend array to NumPy array if necessary.
        """
        if self.use_gpu and isinstance(arr, self.cp.ndarray):
            return self.cp.asnumpy(arr)
        return np.asarray(arr)

    def _rgb_to_od(self, image_rgb):
        """
        Converts an RGB image to Optical Density (OD) space.
        Input: NumPy uint8 array (H,W,3)
        Returns: backend array (float32) with same shape
        """
        xp = self.xp
        img = image_rgb.astype(np.float32) + 1e-8
        # move to backend if needed
        img = self._to_backend(img)
        od = -xp.log10(img / 255.0)
        return od

    def _od_to_rgb(self, image_od):
        """
        Converts an OD image back to RGB space.
        Input: backend array (float) or NumPy array
        Returns: NumPy uint8 array
        """
        xp = self.xp
        od = image_od
        # ensure it's backend array for operations
        if not (self.use_gpu and isinstance(od, self.cp.ndarray)):
            od = self._to_backend(od)

        rgb = (255 * (10 ** (-od))).clip(0, 255)
        rgb_np = self._to_numpy(rgb).astype(np.uint8)
        return rgb_np

    def _compute_covariance(self, X):
        """
        Compute covariance matrix for data X shaped (N, D) using backend xp.
        Returns a (D, D) covariance matrix.
        """
        xp = self.xp
        # X is expected to be backend array
        if not (self.use_gpu and isinstance(X, self.cp.ndarray)):
            X = self._to_backend(X)
        # center
        mean = xp.mean(X, axis=0, keepdims=True)
        Xc = X - mean
        n = Xc.shape[0]
        # compute covariance (D x D)
        cov = (Xc.T @ Xc) / (n - 1.0 if n > 1 else 1.0)
        return cov

    def _eigh(self, cov_matrix):
        """
        Compute eigenvalues and eigenvectors for symmetric cov_matrix. Uses CuPy if enabled, otherwise SciPy.
        Returns (eigenvalues, eigenvectors) where eigenvectors columns correspond to eigenvalues.
        """
        if self.use_gpu:
            # use cupy.linalg.eigh
            w, v = self.cp.linalg.eigh(cov_matrix)
            return w, v
        else:
            # convert to NumPy for SciPy
            cov_np = np.asarray(cov_matrix)
            w, v = cpu_eigh(cov_np)
            return w, v

    def _lstsq(self, A, B):
        """
        Solve least squares A x = B for possibly backend arrays.
        A: (D, k) ; B: (D, N) or (D,) — in our usage we pass (3,2) and (3, M)
        Returns solution with shape (k, N) as backend array.
        """
        xp = self.xp
        if self.use_gpu:
            # cupy.linalg.lstsq returns (x, residuals, rank, s)
            x, *_ = self.cp.linalg.lstsq(A, B)
            return x
        else:
            x, *_ = np.linalg.lstsq(np.asarray(A), np.asarray(B), rcond=None)
            return x

    def _percentile(self, arr, q):
        xp = self.xp
        # backend percentile
        if self.use_gpu:
            return float(self.cp.percentile(arr, q))
        else:
            return float(np.percentile(arr, q))

    def _get_stain_vectors_and_concentrations(self, image_rgb):
        """
        Estimates stain vectors (H&E) and their concentrations from an RGB image.
        Returns: (stain_vectors (3x2) as NumPy array, concentrations (N_pixels x 2) as NumPy array, original_shape)
        """
        xp = self.xp
        cp = self.cp

        # 1. Convert RGB to OD (backend array)
        od = self._rgb_to_od(image_rgb)
        original_shape = tuple(map(int, od.shape))
        od_reshaped = od.reshape(-1, 3)  # backend array shape (N, 3)

        # 2. Filter out 'white' or 'background' pixels (sum OD small)
        # sum along color channels
        sums = xp.sum(od_reshaped, axis=1)
        mask = sums > 0.15
        # If too few pixels pass the mask, relax threshold
        n_pass = int(mask.sum()) if self.use_gpu else int(np.sum(mask))
        if n_pass < 10:
            mask = sums > 0.01

        od_filtered = od_reshaped[mask]

        # 3. Compute covariance on filtered OD and get principal components
        cov_matrix = self._compute_covariance(od_filtered)

        # eigen decomposition
        eigenvalues, eigenvectors = self._eigh(cov_matrix)

        # Sort eigenvalues/eigenvectors descending
        # For backend arrays, move to numpy for sorting indices if using GPU
        if self.use_gpu:
            eigvals_np = self.cp.asnumpy(eigenvalues)
            idx = eigvals_np.argsort()[::-1]
            eigenvectors = eigenvectors[:, idx]
        else:
            idx = eigenvalues.argsort()[::-1]
            eigenvectors = eigenvectors[:, idx]

        v = eigenvectors[:, :2]  # shape (3,2)

        # Ensure components point to positive quadrant for consistency
        # operate in backend if possible
        if self.use_gpu:
            for i in range(v.shape[1]):
                col = v[:, i]
                if float(self.cp.sum(col)) < 0:
                    v[:, i] = -v[:, i]
        else:
            for i in range(v.shape[1]):
                if np.sum(v[:, i]) < 0:
                    v[:, i] *= -1

        # Normalize stain vectors (columns)
        if self.use_gpu:
            norms = self.cp.linalg.norm(v, axis=0)
            stain_vectors = v / norms
            # project filtered OD onto v for angle calc
            proj = od_filtered @ v
            angles = self.cp.arctan2(proj[:, 1], proj[:, 0])
            min_angle = float(self.cp.percentile(angles, self.percentiles[0]))
            max_angle = float(self.cp.percentile(angles, self.percentiles[1]))
        else:
            norms = np.linalg.norm(v, axis=0)
            stain_vectors = v / norms
            proj = od_filtered @ v
            angles = np.arctan2(proj[:, 1], proj[:, 0])
            min_angle = float(np.percentile(angles, self.percentiles[0]))
            max_angle = float(np.percentile(angles, self.percentiles[1]))

        # 6.g Project ALL pixels onto stain_vectors to get concentrations
        # Solve stain_vectors (3x2) * C.T (2xN) = od_reshaped.T (3xN) => C = lstsq(stain_vectors, od_reshaped.T).T
        # Ensure inputs to lstsq are backend arrays
        A = stain_vectors  # shape (3,2) backend array
        B = od_reshaped.T  # shape (3, N)
        # Use backend lstsq
        X = self._lstsq(A, B)  # solution shape (2, N) as backend array
        concentrations = X.T  # shape (N, 2)

        # Ensure non-negative
        if self.use_gpu:
            concentrations[concentrations < 0] = 0
            stain_vectors_np = self.cp.asnumpy(stain_vectors)
            concentrations_np = self.cp.asnumpy(concentrations)
        else:
            concentrations[concentrations < 0] = 0
            stain_vectors_np = np.asarray(stain_vectors)
            concentrations_np = np.asarray(concentrations)

        return stain_vectors_np, concentrations_np, original_shape

    def get_mean_reference_stain_characteristics(self, list_of_reference_images_rgb):
        """
        Computes mean stain vectors and mean max concentrations from a list of reference RGB images.
        Returns (mean_ref_stain_vectors (3x2 NumPy), (mean_max_h, mean_max_e)).
        """
        if not list_of_reference_images_rgb:
            raise ValueError("list_of_reference_images_rgb cannot be empty.")

        all_stain_vectors = []
        all_max_h = []
        all_max_e = []

        for img in list_of_reference_images_rgb:
            v, conc, _ = self._get_stain_vectors_and_concentrations(img)
            all_stain_vectors.append(v)
            all_max_h.append(np.percentile(conc[:, 0], self.percentiles[1]))
            all_max_e.append(np.percentile(conc[:, 1], self.percentiles[1]))

        mean_ref_stain_vectors = np.mean(np.stack(all_stain_vectors, axis=0), axis=0)
        mean_max_h = float(np.mean(all_max_h))
        mean_max_e = float(np.mean(all_max_e))
        return mean_ref_stain_vectors, (mean_max_h, mean_max_e)

    def normalize(self, target_image_rgb, reference_image_rgb=None,
                  mean_ref_stain_vectors=None, mean_ref_max_concentrations_tuple=None):
        """
        Normalize target_image_rgb to match reference stain characteristics.
        Either provide a single reference image via reference_image_rgb, or provide
        mean_ref_stain_vectors and mean_ref_max_concentrations_tuple (mean_max_h, mean_max_e).

        Returns a NumPy uint8 RGB image.
        """
        # validate reference inputs
        provided = 0
        if reference_image_rgb is not None:
            provided += 1
        if mean_ref_stain_vectors is not None and mean_ref_max_concentrations_tuple is not None:
            provided += 1
        if provided == 0:
            raise ValueError("Either reference_image_rgb or (mean_ref_stain_vectors and mean_ref_max_concentrations_tuple) must be provided.")
        if provided > 1:
            raise ValueError("Provide only one form of reference (single image OR mean reference tuple).")

        if reference_image_rgb is not None:
            ref_v, ref_conc, _ = self._get_stain_vectors_and_concentrations(reference_image_rgb)
            mean_ref_max_h = float(np.percentile(ref_conc[:, 0], self.percentiles[1]))
            mean_ref_max_e = float(np.percentile(ref_conc[:, 1], self.percentiles[1]))
            ref_stain_vectors = ref_v
        else:
            ref_stain_vectors = np.asarray(mean_ref_stain_vectors)
            mean_ref_max_h, mean_ref_max_e = mean_ref_max_concentrations_tuple

        # get target concentrations
        target_v, target_conc, target_shape = self._get_stain_vectors_and_concentrations(target_image_rgb)

        max_target_h = float(np.percentile(target_conc[:, 0], self.percentiles[1]))
        max_target_e = float(np.percentile(target_conc[:, 1], self.percentiles[1]))

        # scale concentrations
        scaling_h = mean_ref_max_h / (max_target_h + 1e-8)
        scaling_e = mean_ref_max_e / (max_target_e + 1e-8)

        normalized_conc = np.copy(target_conc)
        normalized_conc[:, 0] = target_conc[:, 0] * scaling_h
        normalized_conc[:, 1] = target_conc[:, 1] * scaling_e
        normalized_conc[normalized_conc < 0] = 0

        # reconstruct OD using reference stain vectors: OD = C_normalized @ V_ref.T
        od_norm_reshaped = normalized_conc @ ref_stain_vectors.T
        od_norm = od_norm_reshaped.reshape(target_shape)

        # convert back to RGB (NumPy uint8)
        rgb_norm = self._od_to_rgb(od_norm)
        return rgb_norm
