import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


def grad_energy(img, sigma = 3, rescale=255):
    """
    Compute the gradient magnitude of an image by doing
    1D convolutions with the derivative of a Gaussian
    
    Parameters
    ----------
    img: ndarray(M, N, 3)
        A color image
    sigma: float
        Width of Gaussian to use for filter
    scale: float
        Amount by which to rescale the gradient
        
    Returns
    -------
    ndarray(M, N): Gradient Image
    """
    I = 0.2125*img[:, :, 0] + 0.7154*img[:, :, 1] + 0.0721*img[:, :, 2]
    I = I/255
    N = int(sigma*6+1)
    t = np.linspace(-3*sigma, 3*sigma, N)
    dgauss = -t*np.exp(-t**2/(2*sigma**2))
    IDx = convolve2d(I, dgauss[None, :], mode='same')
    IDy = convolve2d(I, dgauss[:, None], mode='same')
    GradMag = np.sqrt(IDx**2 + IDy**2)
    return rescale*GradMag


def plot_seam(img, seam):
    """
    Plot a seam on top of the image
    Parameters
    ----------
    I: ndarray(nrows, ncols, 3)
        An RGB image
    seam: ndarray(nrows, dtype=int)
        A list of column indices of the seam from
        top to bottom
    """
    plt.imshow(img)
    X = np.zeros((len(seam), 2))
    X[:, 0] = np.arange(len(seam))
    X[:, 1] = np.array(seam, dtype=int)
    plt.plot(X[:, 1], X[:, 0], 'r')

def read_image(path):
    """
    A wrapper around matplotlib's image loader that deals with
    images that are grayscale or which have an alpha channel

    Parameters
    ----------
    path: string
        Path to file
    
    Returns
    -------
    ndarray(M, N, 3)
        An RGB color image in the range [0, 255]
    """
    img = plt.imread(path)
    if np.issubdtype(img.dtype, np.integer):
        img = np.array(img, dtype=float)/255
    if len(img.shape) == 3:
        if img.shape[1] > 3:
            # Cut off alpha channel
            img = img[:, :, 0:3]
    if img.size == img.shape[0]*img.shape[1]:
        # Grayscale, convert to rgb
        img = np.concatenate((img[:, :, None], img[:, :, None], img[:, :, None]), axis=2)
    return img

def get_optimal_seam(E):
    """
    Find the optimal seam in a gradient energy map

    Parameters
    ----------
    E: ndarray(M, N)
        A gradient energy map
    
    Returns
    -------
    seam: ndarray(M, dtype=int)
        A list of column indices of the seam from
        top to bottom
    """
    C = np.zeros(E.shape)
    C[0, :] = E[0, :]
    B = np.zeros(E.shape, dtype=int) # Backpointers
    Choices = np.inf*np.ones((3, C.shape[1])) # Choices array; left/center/right
    for i in range(1, E.shape[0]):
        Choices[0, 1::] = C[i-1, 0:-1]
        Choices[1, :] = C[i-1, :]
        Choices[2, 0:-1] = C[i-1, 1::]
        C[i, :] = np.min(Choices, axis=0) + E[i, :]
        B[i, :] = np.argmin(Choices, axis=0)
    j = np.argmin(C[-1, :])
    seam = [j]
    for i in range(C.shape[0]-1, 0, -1):
        j += [-1, 0, 1][B[i, j]]
        seam.append(j)
    seam.reverse()
    return seam

def remove_seam(image, seam):
    '''
    Removes a seam from an image
    
    Parameters
    ----------
    image: ndarray(M, N, 3)
        An RGB image
        
    seam: ndarray(M, dtype=int)
        A list of column indices of the seam from
        top to bottom
    
    Returns
    -------
    ndarray(M, N-1, 3)
        An RGB image with the seam removed
    '''
    
    height, width = image.shape[:2]
    new_image = np.zeros((height, width-1, 3))
    
    for i,j in enumerate(seam):
        new_image[i,0:j] = image[i,0:j]
        new_image[i,j:width] = image[i,j+1:width]
        
    return new_image
        

def mask_erasure(image, mask):
    '''
    Takes in an image and a mask and returns a gradient energy image
    with the mask removed
    
    Parameters
    ----------
    image: ndarray(M, N, 3)
        An RGB image
    
    mask: ndarray(M, N, 3)
        A mask image with the same dimensions as the image
        
    Returns
    -------
    ndarray(M, N)
        A gradient energy image 
    '''
    G = grad_energy(image)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if np.array_equal(mask[i,j], [0,0,0]):
                G[i,j] = 0
    return G

if __name__ == '__main__':
    img2 = read_image("images/penguins_stipple.png")
    G = grad_energy(img2)
    #img3 = read_image("Mask.png")
    #G2 = mask_erasure(img2, img3)
    #opt_num_seams = get_optimal_seam(img2)
    for i in range(1, 100):
        opt = get_optimal_seam(G)
        img2 = remove_seam(img2, opt)
        G = grad_energy(img2)
        #G2 = mask_erasure(img2)
        #opt_num_seams = get_optimal_seam(G2)
        
    plt.imshow(img2)
    plt.savefig("penguin3.png", bbox_inches='tight')

