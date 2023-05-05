# Fast-non-uniform-image-blurring-with-arbitrary-PSF
Efficient method adopted from Astrophysics for image blurring with arbitrary PSF distribution
For mathematical description on algorithm, see T. R. Lauer, “Deconvolution with a spatially-variant PSF,” in Proc. SPIE 4847, Astronomical Data Analysis II, (2002), pp. 167–173.
Simply put, the user can arbitrarily define a spatial-variant MTF (interchangeable with the term "SFR", and is equivalent to PSF) distribution, SFR(k, x, y), of an incoherent imaging system, and observe the output image given an input scene (k is spatial freq., x and y are coordinates in the image space). The method orthogonalizes the spatial-variant PSF and uses global image operations for image blurring, rather than repetitive regional PSF convolution over the entire images. This allows an order of magnitude faster image processing.

Here we describe few parameters in the demo script in more details:
1. imager_SFR: a 3D array of shape (2, num_SFR_point, num_ROI) that specifies the spatial frequency in the first dimension, MTF score in the second dimension, and the index of a spatial location in the third dimension.
2. SFR_coordinates: a 2D array of shape (2, num_ROI) specifies the spatial coordinates x and y at each indexed spatial location.
3. SFR_coordinates_map_size: the height and width of the incoherent imaging system

The script provides two examples of non-uniform image blurring: 
1. synthetic eye foveation effect on a portrait scene (strong peripheral blurring)
2. synthetic atmospheric optical aberration effect on astronomical observation (random spatial blurring) 
While the PSF here is assumed to be isotropic, the method can be extended to anisotropic PSF, simulating directional aberrations such as astigmatism and coma.