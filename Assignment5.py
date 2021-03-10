# Note, data for this program should live in /data/celeba
# I have that in gitignore b/c it's too large to hold in github. This program won't work unless you download
# The needed dataset and point data root to it.

from time import time

if __name__ == "__main__":
    print("Start GAN")

    start_time = time()
    time_to_fit = time() - start_time
