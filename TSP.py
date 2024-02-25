import numpy as np
import matplotlib.pyplot as plt
from stipple import *

np.random.seed(0)
I = read_image("images/penguins.png")
# Initial stippling
X = voronoi_stipple(I, thresh=0.3, target_points=2000, canny_sigma=0.8)
# Filter out lowest 4 points by density
X = density_filter(X, (X.shape[0]-4)/X.shape[0])
tour = compute_tour(X)
plt.figure(figsize=(10, 10))
plt.plot(X[tour, 0], X[tour, 1], c='k')
plt.scatter(X[tour, 0], X[tour, 1], s=15,
            c=np.arange(len(tour)), cmap='magma_r')
plt.gca().set_facecolor((0.8, 0.8, 0.8))

plt.savefig("penguins_stipple.png", bbox_inches='tight')
plt.show()
