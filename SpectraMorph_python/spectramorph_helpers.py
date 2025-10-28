# Author: Ritik Shah

import scipy.io as sio
import numpy as np
from tqdm import tqdm
import os
import math
import time
import io as iot
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import tensorflow as tf
from tensorflow.keras.layers import Dense, ReLU, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from utils import downsample_image

def numpy_to_tf(np_array):
    
    """
    Converts a numpy array into a tensorflow tensor.
    """
    
    tf_tensor = tf.constant(np_array, dtype=tf.float32)
    return tf_tensor

def tf_to_numpy(tf_tensor):
    
    """
    Converts a tensorflow tensor into a numpy array.
    """
    
    np_array = tf_tensor.numpy()
    return np_array

def apply_srf_tf(hsi, srf):
    """
    Tensorflow based function to apply a SRF to an image

    Parameters:
        hsi (tf.tensor): The hyperspectral image to which the SRF should be applied of shape (h,w,C)
        srf (tf.tensor): The srf to apply to the image of shape (msi_bands, hsi_bands)

    Returns: msi (tf.tensor): The multispectral image resulting from the application of the srf to the hyperspectral image of shape (h,w,c)
    """
    # Transpose SRF to shape (L_hsi, num_bands)
    srf_t = tf.transpose(srf)  # [L_hsi, num_bands]

    # Tensordot over the last axis of `image` and first axis of `srf_t`
    # Resulting shape = image.shape[:-1] + (num_bands,)
    msi = tf.tensordot(hsi, srf_t, axes=[[-1], [0]])

    return msi

# Function to apply PSF to an image in tensorflow
def apply_psf_tf(image, psf):
    """
    Applies the PSF via depthwise convolution on each spectral band.
    
    Parameters:
        image: tf.Tensor of shape (B, H, W, C)
        psf: np.ndarray of shape (k, k)

    Returns:
        tf.Tensor of shape (B, H, W, C)
    """
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    assert image_tensor.shape.rank == 4, f"Expected 4D input, got {image_tensor.shape}"
    
    # Get number of channels
    C = image_tensor.shape[-1]  # Must be spectral channels, e.g., 191

    # Prepare PSF kernel
    psf_tensor = tf.convert_to_tensor(psf, dtype=tf.float32)
    psf_tensor = tf.reshape(psf_tensor, [*psf_tensor.shape, 1, 1])  # (k, k, 1, 1)
    psf_tensor = tf.tile(psf_tensor, [1, 1, C, 1])  # (k, k, C, 1)

    # Depthwise convolution
    blurred = tf.nn.depthwise_conv2d(
        input=image_tensor,
        filter=psf_tensor,
        strides=[1, 1, 1, 1],
        padding='SAME'
    )
    return blurred

def downsample_image_to_reference_tf(image, reference, method='bicubic'):
    """
    Resize `image` to match the spatial dimensions of `reference` using
    differentiable TensorFlow ops while handling different channel sizes.
    
    Parameters:
    - image: tf.Tensor of shape (H1, W1, C1)
    - reference: tf.Tensor of shape (H2, W2, C2)
    - method: str, one of {'bilinear', 'nearest', 'bicubic', 'area', ...}
    
    Returns:
    - tf.Tensor of shape (H2, W2, C1) (preserves image's channels)
    """
    # Remove batch dimension if present
    if len(image.shape) == 4:
        image = tf.squeeze(image, axis=0)

    if len(reference.shape) == 4:
        reference = tf.squeeze(reference, axis=0)

    # Get spatial dimensions from reference
    target_size = tf.shape(reference)[0:2]

    # Resize with anti-aliasing (preserving input channels)
    resized = tf.image.resize(image, size=target_size, method=method, antialias=True)

    return resized

def plot_abundance_maps(abundance_maps, num_endmembers, return_format="png"):
    """
    Plots abundance maps in a grid and returns the resulting figure as image bytes.

    Parameters
    ----------
    abundance_maps : tf.Tensor or np.ndarray
        A 3D tensor/array of shape (height, width, num_endmembers) containing the abundance
        values for each endmember (component) at every spatial location.

    num_endmembers : int
        The number of endmembers (abundance maps) to visualize. This should match the
        last dimension of `abundance_maps`.

    return_format : str, optional
        The format of the image to return ('png' or 'jpg'). Default is 'png'.

    Returns
    -------
    image_bytes : bytes
        The rendered figure as image bytes (PNG or JPEG).
        These bytes can later be written to a file, e.g.:
            with open("abundance_maps.png", "wb") as f:
                f.write(image_bytes)
    """

    cols = 5
    rows = math.ceil(num_endmembers / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()

    # Compute color scale limits
    vmin = tf.reduce_min(abundance_maps).numpy()
    vmax = tf.reduce_max(abundance_maps).numpy()

    # Plot each abundance map
    for i in range(num_endmembers):
        im = axes[i].imshow(abundance_maps[:, :, i], cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i].set_title(f'Abundance Map {i+1}')
        axes[i].axis('off')

    # Hide unused subplots
    for j in range(num_endmembers, len(axes)):
        axes[j].axis('off')

    # Shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Abundance')

    plt.tight_layout(rect=[0, 0, 0.88, 1])

    # Save to in-memory buffer
    buf = iot.BytesIO()
    fig.savefig(buf, format=return_format, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    image_bytes = buf.getvalue()
    buf.close()

    return image_bytes

def plot_endmember_signatures(endmember_signatures,
                              components=None,
                              normalize=False,
                              title=None,
                              figsize=(11, 6),
                              legend_loc="upper left",
                              return_format="png"):
    """
    Plots spectral signatures for each endmember and returns the resulting figure as image bytes.

    Parameters
    ----------
    endmember_signatures : np.ndarray or tf.Tensor
        Shape (K, C), where K = number of endmembers and C = number of spectral bands/components.

    components : array-like or None, optional
        X-axis values (length C). If None, uses 1..C and labels the axis as 'Components'.

    normalize : bool, optional
        If True, min-max normalizes each signature independently to [0, 1] before plotting.

    title : str or None, optional
        Optional plot title for the figure.

    figsize : tuple, optional
        Size of the figure (width, height). Default is (11, 6).

    legend_loc : str, optional
        Location of the legend (e.g., 'upper left', 'lower right').

    return_format : str, optional
        The format of the returned image ('png' or 'jpg'). Default is 'png'.

    Returns
    -------
    image_bytes : bytes
        The rendered plot as image bytes (PNG or JPEG). These can later be saved using:
            with open("endmember_signatures.png", "wb") as f:
                f.write(image_bytes)
    """
    # --- Convert TensorFlow tensor to NumPy if necessary ---
    try:
        E = endmember_signatures.numpy()
    except AttributeError:
        E = np.asarray(endmember_signatures)

    if E.ndim != 2:
        raise ValueError(f"`endmember_signatures` must be 2D (K, C), got {E.shape}.")

    K, C = E.shape

    # --- X-axis setup ---
    if components is None:
        x = np.arange(1, C + 1)
        x_label = "Components"
    else:
        x = np.asarray(components)
        if x.size != C:
            raise ValueError(f"`components` length ({x.size}) must equal C ({C}).")
        x_label = "Components"

    # --- Optional normalization ---
    if normalize:
        e_min = E.min(axis=1, keepdims=True)
        e_max = E.max(axis=1, keepdims=True)
        E = (E - e_min) / np.maximum(e_max - e_min, 1e-12)

    # --- Plot setup ---
    fig, ax = plt.subplots(figsize=figsize)
    for k in range(K):
        ax.plot(x, E[k], linewidth=2, label=f"Endmember {k+1}")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Intensity")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Legend formatting
    leg = ax.legend(loc=legend_loc, frameon=True, fancybox=False, framealpha=1.0)
    leg.get_frame().set_edgecolor("gray")

    plt.tight_layout()

    # --- Save to in-memory buffer (no GUI required) ---
    buf = iot.BytesIO()
    fig.savefig(buf, format=return_format, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    image_bytes = buf.getvalue()
    buf.close()

    return image_bytes
        
def perform_endmember_extraction(hsi, num_endmembers):
    """
    Performs endmember extraction using Non-negative Matrix Factorization (NMF).

    Parameters
    ----------
    hsi : np.ndarray
        Hyperspectral image cube with shape (H, W, C), where:
            H = image height,
            W = image width,
            C = number of spectral bands.

    num_endmembers : int
        The number of endmembers (distinct material signatures) to extract.

    Returns
    -------
    endmember_signatures : np.ndarray
        Array of shape (num_endmembers, C), where each row is an estimated
        endmember spectral signature.

    Notes
    -----
    - The function uses NMF to decompose the hyperspectral data into an abundance
      matrix (non-negative linear mixture coefficients) and a matrix of
      endmember signatures.
    - The NMF model is initialized using the 'nndsvda' strategy for stable results.
    - The abundance maps themselves are not returned here — only the spectral
      signatures of the endmembers.
    """

    # Clip hyperspectral values to ensure they are within [0, 1]
    # This helps stabilize NMF optimization, which assumes non-negative inputs.
    hsi = np.clip(hsi, 0.0, 1.0)

    # Unpack hyperspectral image dimensions
    H, W, c = hsi.shape

    # Reshape the 3D hyperspectral cube (H, W, C) into a 2D matrix (N, C),
    # where each row represents one pixel's spectral vector.
    V = hsi.reshape(-1, c)

    # Initialize NMF model:
    # - n_components = num_endmembers defines the number of spectral signatures to extract.
    # - init='nndsvda' provides a good non-negative initialization.
    # - max_iter controls convergence limit.
    # - random_state ensures reproducibility.
    model = NMF(n_components=num_endmembers, init='nndsvda', max_iter=5000, random_state=42)

    # Fit NMF to the hyperspectral data (V) to obtain:
    # - W (abundance matrix): pixel-level mixture coefficients
    # - H (components_): spectral signatures (endmembers)
    _ = model.fit_transform(V)

    # Extract endmember spectral signatures (shape: num_endmembers × C)
    endmember_signatures = model.components_

    # Return only the endmember spectra; abundances can be computed separately
    return endmember_signatures

def prepare_inputs(hr_msi, lr_hsi, srf, num_endmembers, spec_prior=False, prior_downsample=4):
    """
    Prepares the data for training.

    Parameters:
        hr_msi (np.ndarray): The high spatial resolution multispectral image of shape (H,W,c)
        lr_hsi (np.ndarray): The low spatial resolution hyperspectral image of shape (h,w,C)
        srf (np.ndarray): The spectral response function of the MSI sensor (can be approximated gaussians) of shape (msi_bands, hsi_bands)
        spec_prior (boolean): A flag indicating whether the coarse spectral prior should be used
        prior_downsample (int): The downsampling factor to build the coarse spectral prior (only used if spec_prior=True)

    Returns:
        hr_msi (tf.Tensor): The high spatial resolution multispectral image of shape (H,W,c)
        lr_hsi (tf.Tensor): The low spatial resolution hyperspectral image of shape (h,w,C)
        lr_msi (tf.Tensor): The low spatial resolution multispectral image of shape (h,w,c)
        downsampled_lr_hsi (tf.Tensor): The downsampled low spatial resolution coarse spectral prior of shape (h/prior_downsample, w/prior_downsample, C)
        abundance_maps (tf.Tensor): The abundance maps obtained from lr_hsi through NMF based endmember extraction of shape (h,w,num_endmembers)
        endmember_signatures (tf.Tensor): The endmember signatures obtained from lr_hsi through NMF based endmember extraction of shape (num_endmembers, C)
    """

    if spec_prior:
        downsampled_lr_hsi = downsample_image(lr_hsi, prior_downsample)
        downsampled_lr_hsi = numpy_to_tf(downsampled_lr_hsi)
    else:
        downsampled_lr_hsi = None

    # Obtaining the abundance maps and endmember signatures
    endmember_signatures = perform_endmember_extraction(lr_hsi, num_endmembers)

    # Generating LR MSI input and converting to tensorflow tensors    
    hr_msi = numpy_to_tf(hr_msi)
    lr_hsi = numpy_to_tf(lr_hsi)
    srf = numpy_to_tf(srf)
    lr_msi = apply_srf_tf(lr_hsi, srf)
    endmember_signatures = numpy_to_tf(endmember_signatures)

    return hr_msi, lr_msi, lr_hsi, downsampled_lr_hsi, endmember_signatures

def get_gpu_memory_mb():
    """Returns current GPU memory usage (in MB) for GPU:0 using TensorFlow."""
    mem_info = tf.config.experimental.get_memory_info('GPU:0')
    return mem_info['current'] / (1024 ** 2)  # bytes → MB

def infer_and_analyze_model_performance_tf(model, sample_inputs):
    """
    Analyzes model complexity: FLOPs, parameters, inference time, and GPU memory usage.
    
    Parameters:
    - model (tf.keras.Model): The model to evaluate.
    - sample_inputs (list of tf.Tensor): List of input tensors matching the model's expected input.
    """
    # 1) Convert sample_inputs into a concrete function
    inputs = [x for x in sample_inputs if x is not None]
    input_signature = [tf.TensorSpec(shape=inp.shape, dtype=inp.dtype) for inp in inputs]

    # Properly trace the model using a callable
    @tf.function
    def model_fn_1(msi):
        return model(msi, None)                 # prior defaults to None in model

    @tf.function
    def model_fn_2(msi, prior):
        return model(msi, prior)

    if len(inputs) == 1:
        concrete_func = model_fn_1.get_concrete_function(*input_signature)
        run_fn = model_fn_1
    elif len(inputs) == 2:
        concrete_func = model_fn_2.get_concrete_function(*input_signature)
        run_fn = model_fn_2
    else:
        raise ValueError("sample_inputs must be [msi] or [msi, prior].")

    # 2) Freeze the graph
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    graph_def = frozen_func.graph.as_graph_def()

    # 3) Compute FLOPs
    try:
        original_stdout = sys.stdout
        sys.stdout = iot.StringIO()

        with tf.Graph().as_default() as graph:
            tf.compat.v1.import_graph_def(graph_def, name="")
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            opts["output"] = "none"
            flops = tf.compat.v1.profiler.profile(
                graph=graph,
                run_meta=run_meta,
                options=opts
            ).total_float_ops
    finally:
        sys.stdout = original_stdout

    # 4) Count parameters and record starting GPU memory
    num_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
    start_mem = get_gpu_memory_mb()

    # 5) Time inference
    start = time.perf_counter()
    SR_image, abundances = run_fn(*inputs)
    end = time.perf_counter()
    inference_time = end - start

    # 6) GPU memory
    end_mem = get_gpu_memory_mb()
    mem_used = end_mem - start_mem

    return SR_image, abundances, num_params, flops, mem_used, inference_time

def batched_inference(model, msi, prior=None, batch_size=None, pbar_desc="Inference"):
    """
    Runs model(msi[, prior]) over spatial tiles and stitches the outputs.
    Expects model to return (sr, abundances) Tensors.

    msi   : np.ndarray | tf.Tensor, shape (H, W, C_msi)
    prior : None | np.ndarray | tf.Tensor, shape (H, W, C_prior) aligned with msi
    batch_size : int | None  (tile size); if None runs full image
    """
    # Convert inputs to NumPy for easy slicing
    if isinstance(msi, tf.Tensor):   msi   = msi.numpy()
    if isinstance(prior, tf.Tensor): prior = prior.numpy() if prior is not None else None

    H, W, _ = msi.shape
    bs = batch_size or max(H, W)
    ys = list(range(0, H, bs))
    xs = list(range(0, W, bs))

    SR_full = None
    AB_full = None

    pbar = tqdm(total=len(ys) * len(xs), desc=pbar_desc)
    for y0 in ys:
        y1 = min(H, y0 + bs)
        for x0 in xs:
            x1 = min(W, x0 + bs)

            patch_msi   = tf.constant(msi[y0:y1, x0:x1, :], dtype=tf.float32)
            patch_prior = None if prior is None else tf.constant(prior[y0:y1, x0:x1, :], dtype=tf.float32)

            sr_patch, ab_patch = model(patch_msi, patch_prior)  # -> Tensors
            sr_patch = sr_patch.numpy()
            ab_patch = ab_patch.numpy()

            # Lazy allocate output arrays using channel dims from first patch
            if SR_full is None:
                SR_full = np.zeros((H, W, sr_patch.shape[-1]), dtype=sr_patch.dtype)
            if AB_full is None:
                AB_full = np.zeros((H, W, ab_patch.shape[-1]), dtype=ab_patch.dtype)

            SR_full[y0:y1, x0:x1, :] = sr_patch
            AB_full[y0:y1, x0:x1, :] = ab_patch

            pbar.update(1)
    pbar.close()
    return SR_full, AB_full

class MSItoHSI_MLP(Model):
    def __init__(self, endmember_signatures, hidden_size=1024):
        """
        Initializes a small MLP model that estimates abundances from MSI and reconstructs HSI using fixed endmember signatures.

        Parameters:
            endmember_signatures (np.ndarray or tf.Tensor): Matrix of shape (K, C_hsi)
            hidden_size (int): Number of units in each hidden layer
        """
        super(MSItoHSI_MLP, self).__init__()
        
        # Store fixed endmember signatures E (K, C_hsi)
        E = tf.convert_to_tensor(endmember_signatures, dtype=tf.float32)
        self.E = tf.constant(E, dtype=tf.float32)
        self.K = int(E.shape[0])
        self.C_hsi = int(E.shape[1])
        
        # Define the MLP layers explicitly
        self.layer1 = Dense(hidden_size, activation='relu', dtype=tf.float32)
        
        # Abundance layer (K outputs, softmax for sum-to-one)
        self.abundance_layer = Dense(self.K, activation='linear', dtype=tf.float32)

    def call(self, msi, prior_hsi):
        """
        Forward pass of the model.

        Parameters:
            msi (tf.Tensor): MSI image of shape (H, W, c_msi)
            prior_hsi (tf.Tensor): Downsampled lr hsi to construct Coarse Spectral Prior for training from

        Returns:
            hsi_hat (tf.Tensor): Reconstructed HSI image of shape (H, W, C_hsi)
        """
        # 1) Grab dynamic sizes
        H = tf.shape(msi)[0]
        W = tf.shape(msi)[1]

        # 2) Flatten inputs
        msi = tf.reshape(msi, [H * W, -1])

        if prior_hsi is not None:
            prior_hsi = tf.reshape(prior_hsi, [H * W, -1])
            x = tf.concat([msi, prior_hsi], axis=-1) # [batch, c+C]
        else:
            x = msi

        # 3) Forward through MLP
        x = self.layer1(x)

        # 4) Estimate abundances
        abundances = self.abundance_layer(x)  # (H*W, K)

        # 5) Linear mixing using endmember signatures E
        hsi_hat_flat = tf.matmul(abundances, self.E)  # (H*W, C_hsi)

        # 6) Reshape outputs
        hsi_hat = tf.reshape(hsi_hat_flat, [H, W, self.C_hsi])     # (H, W, C_hsi)
        abundances = tf.reshape(abundances, [H, W, self.K])        # (H, W, K)

        return hsi_hat, abundances

def train_mlp(
    lr_msi, 
    lr_hsi, 
    lr_lr_hsi, 
    endmember_signatures, 
    epochs, 
    init_lr, 
    max_lr, 
    final_lr, 
    hidden_size,
    batch_size=1024):
    """
    Trains the MLP to estimate abundance maps from msi inputs per pixel.

    Parameters:
        lr_msi (tf.Tensor): Low-res MSI (h,w,c)
        lr_hsi (tf.Tensor): Low-res HSI (h,w,C)
        lr_lr_hsi (tf.Tensor): Downsampled Low res HSI (h/prior_downsample, w/prior_downsample, C) if using Coarse Spectral Prior else None
        endmember_signatures (tf.Tensor): Endmember signatures obtained from NMF (c,C)
        epochs (int): Total number of epochs
        init_lr (float): Initial learning rate
        max_lr (float): Peak learning rate
        final_lr (float): Learning rate at end of training
        hidden_size (int): Number of hidden units in MLP
        batch_size (int): Spatial tile size (B), if None, trains on the full image

    Returns:
        trained_model (tf.keras.Model): Trained MLP model to estimate abundance maps from msi
    """

    # Instantiating the model
    model = MSItoHSI_MLP(endmember_signatures=endmember_signatures, hidden_size=hidden_size)

    H, W, c = lr_msi.shape
    _, _, C = lr_hsi.shape
    
    # Compute prior if given
    if lr_lr_hsi is not None:
        ratio = H // lr_lr_hsi.shape[0]
        x, y = tf.range(W), tf.range(H)
        xx, yy = tf.meshgrid(x, y)
        a = tf.clip_by_value(tf.math.floordiv(yy, ratio), 0, tf.shape(lr_lr_hsi)[0] - 1)
        b = tf.clip_by_value(tf.math.floordiv(xx, ratio), 0, tf.shape(lr_lr_hsi)[1] - 1)
        coords = tf.stack([a, b], axis=-1)
        prior_hsi = tf.gather_nd(lr_lr_hsi, coords)
    else:
        prior_hsi = None

    # Defining the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr)
    # Defining the loss function
    mae = tf.keras.losses.MeanAbsoluteError()

    # Define One-Cycle LR schedule
    def one_cycle_lr(epoch, total_epochs, init_lr, max_lr, final_lr, pct_up=0.3):
        if epoch < pct_up * total_epochs:
            # Linear ramp-up
            return init_lr + (max_lr - init_lr) * (epoch / (pct_up * total_epochs))
        else:
            # Linear ramp-down
            return max_lr - (max_lr - final_lr) * ((epoch - pct_up * total_epochs) / ((1 - pct_up) * total_epochs))

    # Training loop
    pbar = tqdm(range(1, epochs + 1), desc="Training Model", unit="epoch")
    for epoch in pbar:
        current_lr = one_cycle_lr(epoch, epochs, init_lr, max_lr, final_lr)
        optimizer.learning_rate.assign(current_lr)

        # full-image training
        if batch_size is None:
            with tf.GradientTape() as tape:
                pred_lr_hsi, estimated_abundances = model(lr_msi, prior_hsi)
                loss = mae(lr_hsi, pred_lr_hsi)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            pbar.set_postfix(loss=f"{loss.numpy():.4f}", lr=f"{current_lr:.2e}")

        # tiled/batched training
        else:
            epoch_loss = 0.0
            count = 0

            # Iterate over non-overlapping spatial tiles (ragged edges handled by slicing)
            for y0 in range(0, H, batch_size):
                y1 = min(H, y0 + batch_size)
                for x0 in range(0, W, batch_size):
                    x1 = min(W, x0 + batch_size)

                    sub_msi = lr_msi[y0:y1, x0:x1, :]
                    sub_hsi = lr_hsi[y0:y1, x0:x1, :]

                    # Slice the precomputed prior for the current tile (if available)
                    sub_prior = None if prior_hsi is None else prior_hsi[y0:y1, x0:x1, :]

                    with tf.GradientTape() as tape:
                        pred_lr_hsi, estimated_abundances = model(sub_msi, sub_prior)
                        loss = mae(sub_hsi, pred_lr_hsi)

                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                    epoch_loss += float(loss.numpy())
                    count += 1

            # report average patch loss
            avg_loss = epoch_loss / max(count, 1)
            pbar.set_postfix(average_loss=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}")

    return model

def run_pipeline(
    HR_MSI,
    LR_HSI,
    srf,
    num_endmembers=5,
    epochs=1000,
    init_lr=1e-4,
    max_lr=1e-2,
    final_lr=1e-6,
    hidden_size=32,
    spec_prior=False, 
    prior_downsample=4,
    training_batch_size=None,
    inference_batch_size=None,
    synthetic=False,
    return_format='png'
):

    """
    Runs the entire SpectraMorph pipeline end to end.
    """

    # Start training timing
    start_time = time.perf_counter()

    # Obtaining the tensorflow tensors of all the required inputs
    hr_msi, lr_msi, lr_hsi, lr_lr_hsi, endmember_signatures = prepare_inputs(
        HR_MSI, LR_HSI, srf, num_endmembers, spec_prior=spec_prior, prior_downsample=prior_downsample
    )

    # Train the MLP on low resolution input-target pair to learn to estimate abundances from msi per pixel
    trained_model = train_mlp(
        lr_msi, lr_hsi, lr_lr_hsi, endmember_signatures, epochs, init_lr, max_lr, final_lr,
        hidden_size, batch_size=training_batch_size
    )

    # End timing
    print(f"Training completed in {time.perf_counter() - start_time:.2f} seconds")

    H, W, c = hr_msi.shape
    _, _, C = lr_hsi.shape

    if spec_prior:
        h, w, _ = lr_hsi.shape
        ratio = H // h
        x, y = tf.range(W), tf.range(H)
        xx, yy = tf.meshgrid(x, y)
        a = tf.clip_by_value(tf.math.floordiv(yy, ratio), 0, tf.shape(lr_hsi)[0] - 1)
        b = tf.clip_by_value(tf.math.floordiv(xx, ratio), 0, tf.shape(lr_hsi)[1] - 1)
        coords = tf.stack([a, b], axis=-1)
        prior = tf.gather_nd(lr_hsi, coords)
    else:
        prior = None

    if inference_batch_size is None:
        print("Inferring on all input pixels...")
        SR_image, abundances, num_params, flops, mem_used, inference_time = infer_and_analyze_model_performance_tf(
            trained_model,
            sample_inputs=[hr_msi, prior]
        )
        print(f"Parameters:      {num_params:,}")
        print(f"FLOPs:           {flops:,}")
        print(f"GPU Memory:      {mem_used:.2f} MB")
        print(f"Inference time:  {inference_time:.4f} sec")
        SR_image = tf_to_numpy(SR_image)
    else:
        print("Inferring on ", inference_batch_size*inference_batch_size, " pixels per batch...")
        SR_image, abundances = batched_inference(
            trained_model, hr_msi, prior=prior, batch_size=inference_batch_size
        )

    endmember_signatures_img = plot_endmember_signatures(endmember_signatures, normalize=False, return_format=return_format)
    alle_img = plot_abundance_maps(abundances, num_endmembers=num_endmembers, return_format=return_format)

    # Ouputting the super resolved image (HR HSI)
    SR_image = np.clip(SR_image, 0, 1)
    
    return SR_image, endmember_signatures_img, alle_img