from typing import List

import pymde
import numpy as np
import pymde.constraints
import torch
import matplotlib.pyplot as plt

from shape_checker import check_shape, check_shapes


@check_shapes(ret=[(None, None), np.float32])
def embed_images(imgs: List[np.ndarray], n_dims=2) -> np.ndarray:
    img_count = len(imgs)
    imgs = np.array(imgs)
    check_shape(imgs, [(img_count, None, None), np.float32])
    imgs = imgs.reshape((img_count, -1))
    check_shape(imgs, [(img_count, None), np.float32])

    # mde = pymde.preserve_neighbors(
    #     torch.Tensor(imgs),
    #     embedding_dim=n_dims,
    #     constraint=pymde.constraints.Standardized(),
    # )
    mde = pymde.preserve_distances(
        torch.Tensor(imgs),
        embedding_dim=n_dims,
        constraint=pymde.constraints.Standardized(),
    )

    embedding = mde.embed()
    print(embedding)

    mde.plot(marker_size=16)
    plt.show()

    return embedding.numpy()
