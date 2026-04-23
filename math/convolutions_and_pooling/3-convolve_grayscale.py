import numpy as np

def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    
    # Handle padding
    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
    else:
        ph, pw = padding
    
    # Pad images with zeros
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    
    # Calculate output dimensions
    padded_h = h + 2 * ph
    padded_w = w + 2 * pw
    output_h = (padded_h - kh) // sh + 1
    output_w = (padded_w - kw) // sw + 1
    
    # Initialize output
    output = np.zeros((m, output_h, output_w))
    
    # Perform convolution with only two loops
    for i in range(output_h):
        for j in range(output_w):
            # Extract patches for all images simultaneously
            patches = padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw]
            # Element-wise multiply with kernel and sum over spatial dimensions
            output[:, i, j] = np.sum(patches * kernel, axis=(1, 2))
    
    return output
