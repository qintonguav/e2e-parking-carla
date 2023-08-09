import numpy as np


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def encode_npy_to_pil(bev_array):
    c, w, h = bev_array.shape

    img = np.zeros([3, w, h]).astype('uint8')
    bev = np.ceil(bev_array).astype('uint8')

    for i in range(c):
        if 0 <= i <= 4:
            # road, lane, light
            img[0] = img[0] | (bev[i] << (8 - i - 1))
        elif 5 <= i <= 9:
            img[1] = img[1] | (bev[i] << (8 - (i - 5) - 1))
        elif 10 <= i <= 14:
            img[2] = img[2] | (bev[i] << (8 - (i - 10) - 1))

    return img
