"""
Starter file for hw6pr2 of Big Data Summer 2017

Before attemping the helper functions, please familiarize with pandas and numpy
libraries. Tutorials can be found online:
http://pandas.pydata.org/pandas-docs/stable/tutorials.html
https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

Please COMMENT OUT any steps in main driver before you finish the corresponding
functions for that step. Otherwise, you won't be able to run the program
because of errors.

Note:
1. When filling out the functions below, note that
        1) Let k be the rank for approximation

2. Please read the instructions and hints carefully, and use the name of the
variables we provided, otherwise, the function may not work.

3. Remember to comment out the TODO comment after you finish each part.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
import imageio
import urllib

if __name__ == '__main__':

        # =============STEP 0: LOADING DATA=================
        # NOTE: Be sure to install Pillow with "pip3 install Pillow"
        print('==> Loading image data...')
        img = imageio.imread('X017qGH.jpg', as_gray=True)

        # TODO: Shuffle the image

        # HINT:
                # 1) Use np.random.shuffle(img) to shuffle an image img
                # 2) np.random.shuffle() only shuffle along the major axis (row).
                #       Be sure to flatten the image with img.flatten() before doing the shuffling

        "*** YOUR CODE HERE ***"
        shuffle_img = img.copy().flatten()
        np.random.shuffle(shuffle_img)

        # used hw6_sol/hw6pr2_sol.py as reference to understand what was desired for shuffle
        # initially I thought we *wanted* the columnwise shuffle
        
        "*** END YOUR CODE HERE ***"
        # reshape the shuffled image
        shuffle_img = shuffle_img.reshape(img.shape)

        # =============STEP 1: RUNNING SVD ON IMAGES=================
        print('==> Running SVD on images...')

        # TODO: SVD on img and shuffle_img

        # HINT:
        #               1) Use np.linalg.svd() to perform singular value decomposition
        #               2) For the naming of variables, decompose img into U, S, V
        #               3) Decompose shuffle_img into U_s, S_s, V_s

        "*** YOUR CODE HERE ***"
        U, S, V = np.linalg.svd(img)

        U_s, S_s, V_s = np.linalg.svd(shuffle_img)

        "*** END YOUR CODE HERE ***"

        # =============STEP 2: SINGULAR VALUE DROPOFF=================
        print('==> Singular value dropoff plot...')
        k = 100
        plt.style.use('ggplot')
        # TODO: Generate singular value dropoff plot

        # NOTE:
        #               1) Make sure to generate lines with different colors or markers

        "*** YOUR CODE HERE ***"
        plt.plot(np.log(S[:100]),markevery=3)
        plt.plot(np.log(S_s[:100]),color='green',markevery=4)

        "*** END YOUR CODE HERE ***"

        plt.legend(['Original', 'Shuffled'])
        # plot/legend syntax has changed
#        plt.legend((orig_S_plot, shuf_S_plot), \
#                ('original', 'shuffled'), loc = 'best')
        plt.title('Singular Value Dropoff for Clown Image')
        plt.ylabel('singular values')
        plt.savefig('dropoff.png', format='png')
        plt.close()

        # =============STEP 3: RECONSTRUCTION=================
        print('==> Reconstruction with different ranks...')
        rank_list = [2, 10, 25]
        plt.subplot(2, 2, 1)
        plt.imshow(img, cmap='Greys_r')
        plt.axis('off')
        plt.title('Original Image')

        # TODO: Generate reconstruction images for each of the rank values

        # HINT:
        #               1) Use plt.imshow() to display images
        #               2) Set cmap='Greys_r' in imshow() to display grey scale images

        for index in range(len(rank_list)):
                k = rank_list[index]
                r = range(k)
                plt.subplot(2, 2, 2 + index)

                "*** YOUR CODE HERE ***"
                q = U.take(r,axis=1) @ (np.eye(k)*S.take([r]))@ V.take(r,axis=0)
                plt.imshow(q, cmap='Greys_r')
                "*** END YOUR CODE HERE ***"

                plt.title('Rank {} Approximation'.format(k))
                plt.axis('off')

        plt.tight_layout()
        plt.savefig('reconstruction.png', format='png')
        plt.close()
