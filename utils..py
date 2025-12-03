#!/usr/bin/env python
# coding: utf-8



import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from skimage.util import view_as_windows
from scipy import stats
from scipy.signal import convolve2d



def scale_image_into_range(image, min_range =  0, max_range = 1):
    """
    Scale image into range [min_range, max_range]
    """
    img_s = (image - image.min())/(image.max() - image.min()) # scaled to 0-1 range
    return min_range + (max_range - min_range)*img_s


def neighborhood_transform(img, f=np.mean, n_neighbors=8):
    if n_neighbors % 8 != 0:
        raise ValueError("parameter n_neighbors should be a multiple of 8.")

    try:
        m, n = img.shape
        is_color = False
    except ValueError:
        m, n, _ = img.shape
        is_color = True

    out = np.zeros_like(img, dtype=float)
    span = n_neighbors // 8
    span_pp = span + 1

    for i in range(m):
        for j in range(n):
            # clamp window coordinates
            i0 = max(i - span, 0)
            i1 = min(i + span_pp, m)
            j0 = max(j - span, 0)
            j1 = min(j + span_pp, n)

            window = img[i0:i1, j0:j1, ...]

            # apply function
            if is_color:
                window = window.reshape((-1,3))
                val = f(window,axis=0 )
            else:
                val = f(window)

            out[i, j, ...] = val

    return out


def show_images_grid(images,figsize =(10, 10), titles=None):
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    max_h, max_w = max(heights), max(widths)

    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=figsize)

    if len(images) == 1:
        axes = [axes]

    for i, (ax, img) in enumerate(zip(axes, images)):
        scale = img.shape[1] / max_w
        ax.imshow(img)
        ax.set_aspect('equal')
        ax.set_title(titles[i] if titles else f"{img.shape}")
        # Shrink axis according to image size
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * scale, box.height * scale])
        ax.axis("off")
        plt.tight_layout()
    plt.show()



def scale_image(img, sx, sy):
    H, W = img.shape[:2]
    H_out, W_out = int(H * sy), int(W * sx)
    out = np.zeros((H_out, W_out, *img.shape[2:]), dtype=img.dtype)

    y_prime, x_prime = np.meshgrid(np.arange(H_out), np.arange(W_out), indexing='ij')
    # inverse mapping
    x = x_prime / sx
    y = y_prime / sy

    # floor and ceil for bilinear
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)

    # fractional parts
    wx = x - x0
    wy = y - y0
    wx = wx[...,np.newaxis]
    wy = wy[...,np.newaxis]

    # gather 4 neighbors and interpolate
    Ia = img[y0, x0,:]
    Ib = img[y0, x1,:]
    Ic = img[y1, x0,:]
    Id = img[y1, x1,:]

    out = (
        Ia * (1 - wx) * (1 - wy)
        + Ib * wx * (1 - wy)
        + Ic * (1 - wx) * wy
        + Id * wx * wy
    )

    return out

def rotate_image(image, theta=30):
    H, W = image.shape[:2]
    theta = np.deg2rad(theta)
    cx, cy = W / 2, H / 2

    y_prime, x_prime = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    out = np.zeros((H, W, *image.shape[2:]), dtype=image.dtype)

    # Inverse mapping (destination → source)
    x_c = x_prime - cx
    y_c = y_prime - cy
    x =  np.cos(theta)*x_c + np.sin(theta)*y_c + cx
    y = -np.sin(theta)*x_c + np.cos(theta)*y_c + cy

    # Valid pixel mask
    mask = (x >= 0) & (x < W-1) & (y >= 0) & (y < H-1)

    # Floor/ceil for interpolation
    x0 = np.clip(np.floor(x).astype(int), 0, W - 2)
    y0 = np.clip(np.floor(y).astype(int), 0, H - 2)
    x1 = x0 + 1
    y1 = y0 + 1


    Ia = image[y0, x0, ...]
    Ib = image[y0, x1, ...]
    Ic = image[y1, x0, ...]
    Id = image[y1, x1, ...]

    wx = x - x0
    wy = y - y0
    wx = wx[...,np.newaxis]
    wy = wy[..., np.newaxis]

    out = (
        Ia * (1 - wx) * (1 - wy)
        + Ib * wx * (1 - wy)
        + Ic * (1 - wx) * wy
        + Id * wx * wy
    )

    out[~mask] = 0
    return out


def is_pixel_scaled(img):
    return img.max() <= 1

def image_negative(img):
    L = 1 if is_pixel_scaled(img) else 255
    return np.clip(L - img,0,L)

def log_transform(img, c = 1):
    if is_pixel_scaled(img):
        img= scale_image_into_range(img,max_range=255)
    out = c*np.log(img+1)
    return scale_image_into_range(out)

def power_law_transform(img, c = 1, gamma = 1):
    if is_pixel_scaled(img):
        img =  img= scale_image_into_range(img,max_range=255)
    return scale_image_into_range(c*img**gamma)


def sigmoid(x, a = 1):
    return 1 / (1 + np.exp(-a*x))



def contrast(img,*arg,**kws):
    img = scale_image_into_range(img, min_range = -1, max_range=1)
    return scale_image_into_range(sigmoid(img,*arg,**kws))


def bit_plane_decomposition(img, bitsize = 8):
    if is_pixel_scaled(img):
        img = scale_image_into_range(img, max_range=255)

    assert ~(bitsize%2), "parameter bitsize must be multiple of 2"
    out = []
    for i in reversed(range(0,bitsize)):
        divider = 2**i
        out.append(img//divider)
        img %= divider

    return out

def bit_plane_reconst_img(imgs, idx_list,bitsize = 8):
    assert np.max(idx_list) < bitsize, f"idx_list must be between [0, {bitsize})"
    n = bitsize - 1
    out = np.zeros_like(imgs[0])
    for i in idx_list:
        out += imgs[i]*(2**(n-i))
    return out

def histogram_equalization_grayscale(img, nbits=8):
    L = 2**nbits

    if is_pixel_scaled(img):
        img = scale_image_into_range(img, max_range= (L-1))

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

            # Get dimensions
    if img.ndim == 2:
        m, n = img.shape
    else:
        raise ValueError("Input must be a 2D grayscale image.")


    intensity,count = np.unique(img, return_counts = True)
    intensity = intensity.astype(np.uint8)
    d = m*n

    probs = np.zeros(L)
    for p,c in zip(intensity,count):
        probs[p] = c/d
    s = (L-1)*probs.cumsum()
    s = np.clip(s, 0, 255)

    return s[img.astype(np.uint8)]


def box_smoothing(img, kernel_size=3):
    assert kernel_size % 2 == 1, "kernel size must be a positive odd integer"
    pad = kernel_size // 2

    # Ensure float or uint8 input
    img = img.astype(np.float32)

    # Pad image (replicate padding gives better smoothing edges)
    if img.ndim == 2:
        padded = np.pad(img, pad, mode="reflect")
    else:
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

    # Compute integral image (cumulative sum)
    if img.ndim == 2:
        I = padded.cumsum(axis=0).cumsum(axis=1)
    else:
        I = padded.cumsum(axis=0).cumsum(axis=1)

    H, W = img.shape[:2]
    k = kernel_size

    # Indices for the corners of each window (vectorized)
    # top-left, top-right, bottom-left, bottom-right
    tl = I[:-k, :-k]
    tr = I[:-k, k:]
    bl = I[k:, :-k]
    br = I[k:, k:]

    # Sum of each k×k window (vectorized)
    window_sum = br - bl - tr + tl

    # Average
    out = window_sum / (k * k)

    return out


def gaussian_kernel(kernel_size = 5, sigma = 1):
    assert kernel_size%2 ==1, "kernel_size must be postive odd integer"
    k = kernel_size//2
    x = np.arange(-k,k+1)
    g = np.exp(-(x**2) / (2 * sigma * sigma))
    return g/g.sum()   

def convolve_1d(img,kernel, axis):
    k = len(kernel)//2
    pad_width = [(0,0)]*img.ndim
    pad_width[axis] = (k,k)
    padded = np.pad(img, pad_width, mode = "reflect")

    out = np.zeros_like(img)
    for i, w in enumerate(kernel):
        shift = i - k
        out += w * np.take(padded, indices=range(k+shift, k+shift+img.shape[axis]), axis=axis)
    return out


def gaussian_blur(img, size=5, sigma=1.0):
    g = gaussian_kernel(size, sigma)
    tmp = convolve_1d(img, g, axis=1)  # horizontal
    out = convolve_1d(tmp, g, axis=0)  # vertical
    return out


def laplace(img, ktype = 2):
    assert ktype in [1,2], "ktype my be an integer of 1 or 2"

    ksize = 3 # kernel size

    if ktype == 1:
        k = [0,1,0,1,-4,1,0,1,0]
    else:
        k = [1,1,1,1,-8,1,1,1,1]

    p = ksize//2
    if img.ndim == 2:
        padded = np.pad(img, p, mode="reflect")
    else:
        padded = np.pad(img, ((p, p), (p, p), (0, 0)), mode="reflect")

    out = np.zeros_like(img)

    neighbors = [
    padded[0:-2, 0:-2, ...],   # top-left
    padded[0:-2, 1:-1, ...],   # top
    padded[0:-2, 2:, ...],   # top-right

    padded[1:-1, 0:-2, ...],   # left
    padded[1:-1, 1:-1, ...],   # center
    padded[1:-1, 2:, ...],   # right

    padded[2:, 0:-2, ...],   # bottom-left
    padded[2:, 1:-1, ...],   # bottom
    padded[2:, 2:, ...],   # bottom-right
    ]

    for i,n in zip(k,neighbors):
        out += i*n

    return out

def laplace_sharpen(img, *args, **kws):
    filtered_img = laplace(img, *args, **kws)
    L = 1 if is_pixel_scaled(img) else 255
    return np.clip(img - filtered_img,0,L)


def unsharp_masking(img, k = 1, sigma = 7):
    assert k > 0, "k must be positive"

    L = 1 if is_pixel_scaled(img) else 255

    # compute a blurred image
    size = int(sigma*6)
    while size%2 == 0:
        size += 1

    img_b = gaussian_blur(img, size = size, sigma = sigma)
    mask = img - img_b
    return np.clip(mask,0,L), np.clip(img+k*mask, 0,L)

def sobel_smoothing(img, dist_type = 1):
    assert dist_type in [1,2,3], "dist_type must be 1,2,3"

    L = 1 if is_pixel_scaled(img) else 255

    kernel_size = 3
    p = kernel_size//2
    if img.ndim == 2: # not color
        padded = np.pad(img,p,mode = "reflect")
    else:
        padded = np.pad(img, ((p,p),(p,p),(0,0)), mode = "reflect")

    """
    [z1,z2,z3]
    [z4,z5,z6]
    [z7,z8,z9]
    """
    wx = [-1,-2,-1,0,0,0,1,2,1]
    wy = [-1,0,1,-2,0,2,-1,0,1]

    neighbors = [
    padded[0:-2, 0:-2, ...],   # top-left = z1
    padded[0:-2, 1:-1, ...],   # top = z2
    padded[0:-2, 2:, ...],   # top-right = z3

    padded[1:-1, 0:-2, ...],   # left = z4
    padded[1:-1, 1:-1, ...],   # center = z5
    padded[1:-1, 2:, ...],   # right = z6

    padded[2:, 0:-2, ...],   # bottom-left = z7
    padded[2:, 1:-1, ...],   # bottom = z8
    padded[2:, 2:, ...],   # bottom-right = z9
    ]    

    gx = sum(i*n for i,n in zip(wx,neighbors))
    gy = sum(i*n for i,n in zip(wy,neighbors))

    if dist_type == 1:
        sobel_mask = np.abs(gx) + np.abs(gy)
    elif dist_type == 2:
        sobel_mask = (gx**2 + gy**2)**(0.5)
    elif dist_type == 3:
        sobel_mask = np.maximum(gx, gy)
    else:
        raise ValueError("dist_type can only be one of these integers: 1,2,3.")

    return np.clip(sobel_mask,0,L), np.clip(img + sobel_mask, 0,L)


def gaussian_bandpass(img, big_sigma, small_sigma):
    assert big_sigma > small_sigma, "big_sigma must be larger"

    if img.dtype != np.float64:
        img = img.astype(np.float64)

    def compute_kernel_size(sigma):
        k = max(3,int(sigma*6+1))
        return k if k % 2 else k + 1

    big_ksize = compute_kernel_size(big_sigma)
    small_ksize = compute_kernel_size(small_sigma)


    lp1 = gaussian_blur(img,big_ksize,big_sigma)
    lp2 = gaussian_blur(img,small_ksize,small_sigma)

    return lp2 - lp1


def Dft_1D(fx):
    M = len(fx)
    t = np.arange(M)
    u,x= np.meshgrid(t,t, indexing = "ij")
    args = 2*np.pi*u*x/M
    b = np.cos(args) - 1j*np.sin(args)
    return (fx[None,:]*b).sum(axis = 1)

def iDft_1D(fu):
    M = len(fu)
    t = np.arange(M)
    x,u = np.meshgrid(t,t, indexing = 'ij')
    args = 2*np.pi*u*x/M
    b = np.cos(args) + 1j*np.sin(args)
    return (fu[None,:]*b).sum(axis = 1)/M


def DFT_2D(img):
    """
    Fully vectorized 2-D DFT that works for both grayscale and RGB images.
    img: shape (M, N) or (M, N, 3)
    returns complex array of same shape
    """
    img = img.astype(float)

    if img.ndim == 2:
        img = img[..., None]

    M, N, C = img.shape

    # Frequency indices
    u = np.arange(M)[:, None]   # (M,1)
    v = np.arange(N)[:, None]   # (N,1)

    # Spatial indices
    x = np.arange(M)[None, :]   # (1,M)
    y = np.arange(N)[None, :]   # (1,N)

    # Correct exponent matrices (elementwise multiply)
    Wx = np.exp(-2j * np.pi * (u * x) / M)   # (M,M)
    Wy = np.exp(-2j * np.pi * (v * y) / N)   # (N,N)

    # Apply separable DFT: rows → columns
    F = Wx @ img.reshape(M, N*C)
    F = F.reshape(M, N, C)
    F =  Wy @ F

    return F if C > 1 else F[...,0]


def IDFT_2D(img):
    """
    Fully vectorized inverse 2-D DFT that works for both grayscale and RGB images.
    img: shape (M, N) or (M, N, 3)
    returns complex array of same shape
    """

    if img.ndim == 2:
        img = img[..., None]

    M, N, C = img.shape

    # Frequency indices
    u = np.arange(M)[:, None]   # (M,1)
    v = np.arange(N)[:, None]   # (N,1)

    # Spatial indices
    x = np.arange(M)[None, :]   # (1,M)
    y = np.arange(N)[None, :]   # (1,N)

    # IDFT matrices (use elementwise multiply, not @)
    Wx = np.exp(2j * np.pi * (u * x) / M)   # (M,M)
    Wy = np.exp(2j * np.pi * (v * y) / N)   # (N,N)

    # First: IDFT along rows
    F = Wx @ img.reshape(M, N*C)

    # Second: IDFT along columns
    F = F.reshape(M, N, C)
    F = (Wy @ F)/(M*N) 

    return F if C > 1 else F[...,0]


def padImage(img):
    """Simply pad the image to twice the size. Place image in top left corner"""
    M,N = img.shape
    padded = np.zeros((2*M,2*N))
    padded[:M,:N] = img
    return padded.astype(np.float64)


def ideal_filtering_by_energy(img, percent, filtering_type="low", pad_image = True):
    M_orig, N_orig = img.shape

    if pad_image:
        img = padImage(img)

    M, N = img.shape

    F = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(F)**2

    # coordinate grid
    u = np.arange(-M//2, M//2)
    v = np.arange(-N//2, N//2)
    U, V = np.meshgrid(u, v)
    R = np.sqrt(U**2 + V**2)

    # proper continuous-radius binning
    num_bins = max(M, N) // 2
    bins = np.linspace(0, R.max(), num_bins+1)

    radial_power, _ = np.histogram(R, bins=bins, weights=power)

    cumulative = np.cumsum(radial_power)
    total = cumulative[-1]
    target = percent * total

    # find radius threshold
    idx = np.searchsorted(cumulative, target)
    D0 = bins[idx]  # convert bin index to radius

    print("D0 =", D0)
    print(f"percentage of dc = {cumulative[0]/cumulative[-1]}")

    # build mask
    if filtering_type == "low":
        H = (R <= D0).astype(float)
    else:
        H = (R > D0).astype(float)

    F_filtered = F * H
    result = np.fft.ifft2(np.fft.ifftshift(F_filtered))

    return (np.real(result[:M_orig,:N_orig]),H) if pad_image else (np.real(result), H)



def experimenting_with_ideal_filtering(img, alphas = None, filtering_type = "low", pad_image = True):
    if alphas is None:
        alphas = (0.95, 0.96,.97,.98)

    fig, ax = plt.subplots(nrows=2, ncols=len(alphas), figsize=(10, 8))

    for i, a in enumerate(alphas):
        man_ilpf, ilpf = ideal_filtering_by_energy(img, a, filtering_type=filtering_type,pad_image = pad_image)

        # top row: mask
        ax[0, i].imshow(ilpf, cmap='gray', vmin=0, vmax=1)
        ax[0, i].set_title(f'Mask ({a*100:.0f}%)')
        ax[0, i].axis('off')

        # bottom row: filtered image
        ax[1, i].imshow(man_ilpf, cmap='gray')
        ax[1, i].set_title(f'Filtered Image ({a*100:.0f}%)')
        ax[1, i].axis('off')

    plt.tight_layout()
    plt.show()


def GLPF_filter(shape,sigma):
    M, N = shape

    # Frequency coordinate grid centered at (0, 0)
    u = np.arange(-(M // 2), M // 2)
    v = np.arange(-(N // 2), N // 2)
    U, V = np.meshgrid(u, v, indexing="ij")

    # Squared distance from origin
    D2 = U**2 + V**2

    # Gaussian Low-Pass filter
    H = np.exp(-D2 / (2 * sigma * sigma))
    return H


def GLPF(img, sigma, filtering_type = "low", pad_image = True):
    """
    Gaussian Low-Pass Filter (GLPF)

    Parameters
    ----------
    img : 2D numpy array
        Input grayscale image. Will be converted to float64 if needed.

    sigma : float
        Standard deviation of the Gaussian filter in the frequency domain.
        Controls the cutoff frequency:
            - small sigma  → strong blur (more low-pass)
            - large sigma  → weak blur (gentle smoothing)

    Returns
    -------
    filtered_img : 2D numpy array (float64)
        The filtered image in the spatial domain.

    H : 2D numpy array
        Gaussian low-pass frequency response (same size as image).

    Notes
    -----
    Implements the Gaussian LPF described in Gonzalez & Woods:
        H(u, v) = exp( - (u^2 + v^2) / (2 * sigma^2) )
    """

    assert img.ndim == 2, "Image must be a grayscale (2D) array."

    # Ensure correct dtype
    if img.dtype != np.float64:
        img = img.astype(np.float64)

    ## pad image
    if pad_image:
        M,N = img.shape
        padded = np.zeros((2*M,2*N))
        padded[:M,:N] = img
        img_p = padded
    else:
        img_p = img.copy()

    shape = img_p.shape

    if filtering_type == "low":
        H = GLPF_filter(shape,sigma)
    else:
        H = 1 - GLPF_filter(shape,sigma)

    # Shifted FFT of the image
    F = np.fft.fftshift(np.fft.fft2(img_p))

    # Filtering in frequency domain
    G = F * H

    # Inverse FFT (undo shift)
    filtered = np.fft.ifft2(np.fft.ifftshift(G))

    return (np.real(filtered)[:M,:N], H) if pad_image else (np.real(filtered),H)



def experiment_with_gaussian_filtering(img, sigmas = None, filtering_type = "low", pad_image = True):
    if sigmas is None:
        sigmas = (5,25,50,100)

    fig, ax = plt.subplots(nrows=2, ncols=len(sigmas), figsize=(10, 8))

    for i, s in enumerate(sigmas):
        man_glpf, glpf = GLPF(img, s,filtering_type = filtering_type, pad_image = pad_image)

        # top row: mask
        ax[0, i].imshow(glpf, cmap='gray', vmin=0, vmax=1)
        ax[0, i].set_title(f'Gaussian Filter $\\sigma$={s}')
        ax[0, i].axis('off')

        # bottom row: filtered image
        ax[1, i].imshow(255-man_glpf, cmap='gray')
        ax[1, i].set_title(f'Filtered Image $\\sigma$={s}')
        ax[1, i].axis('off')

    plt.tight_layout()
    plt.show()


def butterworth_filter(img,D0,n):
    assert img.ndim == 2, "Image must be grayscale"

    M, N = img.shape

    # Frequency coordinate grid centered at (0,0)
    u = np.arange(-(M // 2), M // 2)
    v = np.arange(-(N // 2), N // 2)
    U, V = np.meshgrid(u, v, indexing="ij")

    # Distance from the origin (radius)
    D = np.sqrt(U**2 + V**2)

    # Butterworth low-pass filter
    H = 1 / (1 + (D / D0)**(2 * n))
    return H

def butterworth_filtering(img, D0, n,filtering_type = "low", pad_image = True):
    """
    Butterworth Low-Pass Filter (BLPF)

    Parameters
    ----------
    img : 2D numpy array
        Grayscale image.
    D0 : float
        Cutoff frequency (radius where H = 1/sqrt(2)).
    n : int
        Butterworth filter order.

    Returns
    -------
    filtered_img : 2D numpy array (real part)
        The filtered image.
    H : 2D numpy array
        The Butterworth filter transfer function.
    """

    M, N = img.shape
    if pad_image:
        img = padImage(img)

    if filtering_type == "low":
        H = butterworth_filter(img,D0,n)
    else:
        H = 1 - butterworth_filter(img,D0,n)

    # Fourier transform of image
    F = np.fft.fftshift(np.fft.fft2(img))

    # Apply the filter
    G = F * H

    # Inverse FFT
    filtered_img = np.fft.ifft2(np.fft.ifftshift(G))

    # Return real part (imaginary part is numerical noise)
    return (np.real(filtered_img[:M, :N]), H) if pad_image else (np.real(filtered_img), H)



def experiment_with_butterworth_filtering(img, D0 = 10, filter_orders = None,filtering_type = "low", pad_image = True):
    if filter_orders is None:
        filter_orders = (1,2,3,4)

    fig, ax = plt.subplots(nrows=2, ncols=len(filter_orders), figsize=(10, 8))

    for i, n in enumerate(filter_orders):
        man_blpf, blpf = butterworth_filtering(img,D0,n,filtering_type = filtering_type, pad_image = pad_image)

        # top row: mask
        ax[0, i].imshow(blpf, cmap='gray', vmin=0, vmax=1)
        ax[0, i].set_title(f'Mask D0 = {D0}, n = {n}')
        ax[0, i].axis('off')

        # bottom row: filtered image
        ax[1, i].imshow(255-man_blpf, cmap='gray')
        ax[1, i].set_title(f'Filtered Image D0 = {D0}, n = {n}')
        ax[1, i].axis('off')

    plt.tight_layout()
    plt.show()


def laplacian_sharpen_frequency(img,alpha = .2):
    """
    Perform Laplacian image sharpening in the frequency domain.

    Implements: g(x,y) = f(x,y) - ∇² f(x,y)

    Parameters:
        img : 2D numpy array (grayscale)

    Returns:
        sharpened : 2D numpy array (real-valued sharpened image)
        H         : Laplacian filter transfer function (for visualization)
    """

    img = img.astype(float)
    assert img.ndim == 2

    M, N = img.shape
    img = scale_image_into_range(img)
    F = np.fft.fftshift(np.fft.fft2(img))

    # normalized frequency grids (cycles per sample)
    k = np.fft.fftshift(np.fft.fftfreq(M))  # length M, values in [-0.5, 0.5)
    l = np.fft.fftshift(np.fft.fftfreq(N))
    K, L = np.meshgrid(l, k)  # note mesh ordering: axis0 -> rows (k)
    # K = horizontal freq (cols), L = vertical freq (rows)
    # compute squared radius in normalized freq (cycles/sample)
    D2 = K**2 + L**2

    # Laplacian frequency response (normalized)
    H = -4 * (np.pi**2) * D2

    # Apply and scale for sharpening
    LapF = H * F

    lapf = np.real(np.fft.ifft2(np.fft.ifftshift(LapF)))
    lapf = scale_image_into_range(lapf, -1,1)
    out = img - alpha * lapf
    return out, H


def high_frequency_emphasis_filtering(img,sigma, k1 = 1, k2 = .2, pad_image = True):

    if pad_image:
        M, N = img.shape
        img_p = padImage(img)
    else: 
        img_p = img.copy()



    H = GLPF_filter(img_p.shape, sigma)

    F = np.fft.fftshift(np.fft.fft2(img_p))

    # High-frequency emphasis filter
    Hfe = k1 + k2 * (1 - H)

    G = Hfe*F # 1-H is the high-frequency emphasis part, k2 is the high-boost, k1 is the unsharp masking part

    g = np.real(np.fft.ifft2(np.fft.ifftshift(G)))

    return g[:M, :N] if pad_image else g


def homomorphic_filter(shape, sigma, gamma_lp = .5, gamma_hp = 1.5, slope = 1):
    M,N = shape
    u = np.arange(-(M//2), M//2)
    v = np.arange(-(N//2), N//2)
    U,V = np.meshgrid(u,v, indexing = "ij")
    D2 = U**2 + V**2
    H = (gamma_hp - gamma_lp)*(1 - np.exp(-slope*D2/(sigma*sigma))) + gamma_lp
    return H

def homomorphic_filtering(img,sigma, gamma_lp = .5, gamma_hp = 1.5, slope = 1, pad_image = True):
    assert img.ndim == 2, "Grayscale images only"
    if pad_image:
        M,N = img.shape
        img_p = padImage(np.log(img))
    else:
        img_p = np.log(img.copy())

    H = homomorphic_filter(img_p.shape,sigma, gamma_lp , gamma_hp , slope )
    F = np.fft.fftshift(np.fft.fft2(img_p))
    G = H*F
    g = np.real(np.fft.ifft2(np.fft.fftshift(G)))
    g = np.exp(g)
    return (g[:M,:N], H) if pad_image else (g, H)


def gaussian_notch_reject_filter(M, N, u0, v0, sigma):
    """
    M,N : filter size (same as FFT grid, i.e. padded image dims)
    u0,v0: notch center offsets in "frequency index units" relative to center.
          e.g. u0=20 means 20 rows below center in the fftshifted grid.
    sigma: width of the Gaussian notch
    returns H (M,N) with values in [0,1] (1 = pass, 0 = reject)
    """
    u = np.arange(-M//2, M//2)
    v = np.arange(-N//2, N//2)
    U, V = np.meshgrid(u, v, indexing='ij')

    G1 = np.exp(-((U - u0)**2 + (V - v0)**2) / (2 * sigma**2))
    G2 = np.exp(-((U + u0)**2 + (V + v0)**2) / (2 * sigma**2))

    H = (1 - G1) * (1 - G2)   # multiplicative notch reject
    return H

def butterworth_notch_reject_filter(M, N, u0, v0, D0, n):
    u = np.arange(-M//2, M//2)
    v = np.arange(-N//2, N//2)
    U, V = np.meshgrid(u, v, indexing='ij')
    D1 = np.sqrt((U - u0)**2 + (V - v0)**2)
    D2 = np.sqrt((U + u0)**2 + (V + v0)**2)
    B1 = 1 / (1 + (D1 / D0)**(2*n))  # lowpass bump at notch
    B2 = 1 / (1 + (D2 / D0)**(2*n))
    # notch reject: 1 - bump, multiplicative to combine
    H = (1 - B1) * (1 - B2)
    return H


def add_salt_and_pepper_noise(img, p_salt = 0.01, p_pepper = 0.01):
    if is_pixel_scaled(img):
        max_value = 1
    else:
        max_value = 255


    noisy = img.copy()
    prob = np.random.rand(*noisy.shape)

    noisy[prob < p_pepper] = max_value
    noisy[prob > (1 - p_salt)] = 0.0

    return noisy


def geometric_mean_filter(img, k=3,eps=1e-12):
    img = img.astype(np.float64)
    patches = view_as_windows(img, (k, k))
    patches = np.clip(patches,0,None)
    prod = np.prod(patches , axis=(-1, -2))
    geom = prod ** (1.0 / (k*k))
    return geom

def harmonic_mean_filter(img, k=3,eps=1e-12):
    img = img.astype(np.float64)
    patches = view_as_windows(img, (k, k))
    denom = np.sum(1.0 / (patches + eps), axis=(-1, -2))
    harm = (k*k) / denom
    return harm

def contra_harmonic_mean_filter(img, k=3, Q=1.5,eps=1e-12):
    img = img.astype(np.float64)
    patches = view_as_windows(img, (k, k))
    patches = np.clip(patches,0,None)
    num = np.sum((eps + patches) ** (Q + 1), axis=(-1, -2))
    den = np.sum((patches+eps) ** Q, axis=(-1, -2))
    return num / den 



def add_random_noise(img,mode = "gaussian", amount = .5, *args, **kws):
    """
    Adds random noise to image.
    mode = gaussian, expon, rayleigh, uniform
    """
    if not is_pixel_scaled(img):
        img = scale_image_into_range(img)
    if mode == "gaussian":
        pdf = stats.norm(*args, **kws)
        noise = pdf.rvs(size = img.shape)
    elif mode == "expon":
        pdf = stats.expon(*args, **kws)
        noise = pdf.rvs(size = img.shape)
    elif mode == "rayleigh":
        pdf = stats.rayleigh(*args, **kws)
        noise = pdf.rvs(size = img.shape)
    elif mode == "uniform":
        pdf = stats.uniform(*args, **kws)
        noise = pdf.rvs(size = img.shape)
    else:
        raise ValueError("mode must be one of:gaussian, expon, rayleigh, uniform")

    return img + amount*noise


def order_statics_filtering(img, mode="median", k=3, alpha=2):
    """
    Apply order-statistics filtering on an image using k×k neighborhoods.

    Parameters
    ----------
    img : 2D ndarray
        Input grayscale image.
    mode : str
        One of:
            - "median"   : median filter
            - "min"      : min filter
            - "max"      : max filter
            - "midpoint" : (min + max) / 2
            - "alphatrim": alpha-trimmed mean
    k : int
        Neighborhood size (must be odd).
    alpha : int
        Number of values to trim from each side for alpha-trimmed mean.
        Total trimmed = 2*alpha.

    Returns
    -------
    filtered : 2D ndarray
        Output filtered image of shape (M - k + 1, N - k + 1)
    """
    assert k % 2 == 1, "k must be odd."

    img = img.astype(np.float64, copy=False)

    # Extract k×k patches (vectorized sliding windows)
    patches = view_as_windows(img, (k, k))
    # patches.shape = (M-k+1, N-k+1, k, k)

    if mode == "median":
        return np.median(patches, axis=(-1, -2))

    elif mode == "min":
        return np.min(patches, axis=(-1, -2))

    elif mode == "max":
        return np.max(patches, axis=(-1, -2))

    elif mode == "midpoint":
        mins = np.min(patches, axis=(-1, -2))
        maxs = np.max(patches, axis=(-1, -2))
        return 0.5 * (mins + maxs)

    elif mode == "alphatrim":
        # Flatten patches: (M', N', k*k)
        flat = patches.reshape(patches.shape[0], patches.shape[1], -1)

        # Sort along last axis
        sorted_vals = np.sort(flat, axis=-1)

        # Remove `alpha` lowest and `alpha` highest values
        trimmed = sorted_vals[..., alpha : -alpha]

        # Mean of trimmed values
        return np.mean(trimmed, axis=-1)

    else:
        raise ValueError("Invalid mode. Must be one of median, min, max, midpoint, alphatrim.")


def convolve(img, kernel):
    return convolve2d(img, kernel, mode="same", boundary="symm")

def adaptive_local_noise_reduction_filtering(img, k=3, noise_var=None):
    if img.dtype != np.float64:
        img = img.astype(np.float64)

    # Estimate noise variance if not provided (simple Laplacian estimate)
    if noise_var is None:
        lap = np.array([[0, -1,  0],
                        [-1, 4, -1],
                        [0, -1,  0]], dtype=float)
        noise_var = np.mean(np.abs(convolve(img, lap))) / np.sqrt(6)

    patches = view_as_windows(img, (k, k))

    # Local statistics
    local_mean = np.mean(patches, axis=(-1, -2))
    local_var = np.var(patches, ddof=1, axis=(-1, -2))

    # Extract center pixels
    offset = k // 2
    g = patches[:, :, offset, offset]

    # Avoid division by zero
    eps = 1e-8
    local_var = np.maximum(local_var, eps)

    # ALNR factor
    factor = noise_var / local_var
    factor = np.minimum(factor, 1.0)

    # Apply filter
    f_hat = g - factor * (g - local_mean)

    return f_hat



def adaptive_median_filter(img, Smax=7):
    """
    Adaptive Median Filtering for salt-and-pepper noise.
    Implements algorithm from Gonzalez & Woods.

    Parameters
    ----------
    img : 2D ndarray
        Grayscale input image.
    Smax : int
        Maximum window size (must be odd).

    Returns
    -------
    out : 2D ndarray
        Filtered image.
    """

    img = img.astype(np.float64)
    M, N = img.shape
    out = np.zeros_like(img)

    # pad image so windows don't go out of bounds
    pad_size = Smax // 2
    padded = np.pad(img, pad_size, mode='reflect')

    for i in range(M):
        for j in range(N):
            S = 3  # initial window size

            while True:
                half = S // 2
                # extract SxS neighborhood
                patch = padded[i:i+S, j:j+S]

                Zmin = patch.min()
                Zmax = patch.max()
                Zmed = np.median(patch)
                Zxy = img[i, j]

                A1 = Zmed - Zmin
                A2 = Zmed - Zmax

                if A1 > 0 and A2 < 0:
                    # Go to Stage B
                    B1 = Zxy - Zmin
                    B2 = Zxy - Zmax

                    if B1 > 0 and B2 < 0:
                        out[i, j] = Zxy   # keep original
                    else:
                        out[i, j] = Zmed  # replace w/ median
                    break

                else:
                    # Increase window size
                    S += 2
                    if S > Smax:
                        out[i, j] = Zmed
                        break

    return out


def add_periodic_noise(img, A=30, u0=5, v0=0):
    """
    Adds periodic sinusoidal noise to an image.

    A  = amplitude of noise
    u0 = horizontal frequency
    v0 = vertical frequency
    """
    M, N = img.shape
    x = np.arange(M)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y, indexing='ij')

    noise = A * np.sin(2*np.pi * (u0*X/M + v0*Y/N))
    return img + noise


def fourier_power_spectrum(img):
    F = np.fft.fftshift(np.fft.fft2(img))
    return np.abs(F)**2

def log_power_spectrum(img):
    F = np.fft.fftshift(np.fft.fft2(img))
    return np.log(1 + np.abs(F)**2)


