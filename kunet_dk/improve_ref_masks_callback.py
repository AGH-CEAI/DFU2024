import keras
import numpy as np

from skimage import color

from kunet_dk.unetlike import Unetlike
from utils import norm_img


class ImproveRefMasksCallback(keras.callbacks.Callback):
    model_initial_weights_path = './model_initial_weights.h5'
    def __init__(self, model, eval_step_ratio, network_input_wh,
                 train_images, train_masks, val_imgs, val_masks,
                 mask_weaken_modifier, mask_weaken_modifier_decay,
                 patience, patience_increase):
        super().__init__()
        self._net: Unetlike = model
        self._eval_step_ratio = eval_step_ratio
        self._network_input_wh = network_input_wh
        self._train_images = train_images
        self._train_masks = train_masks
        self._val_imgs = val_imgs
        self._val_masks = val_masks
        self._mask_weaken_modifier = mask_weaken_modifier+mask_weaken_modifier_decay
        self._mask_weaken_modifier_decay = mask_weaken_modifier_decay

        self._patience = patience
        self._patience_increase = patience_increase

        self._epochs_from_last_model_improvement = 0
        self._best_val_loss = np.inf

        self._net.model.save_weigths(self.model_initial_weights_path)

    def on_epoch_end(self, epoch, logs=None):
        if logs['val_loss'] < self._best_val_loss:
            self._best_val_loss = logs['val_loss']
            self._epochs_from_last_model_improvement = 0
            return

        self._epochs_from_last_model_improvement += 1
        if self._epochs_from_last_model_improvement < self._patience:
            return

        print(f'\n{self._epochs_from_last_model_improvement} epochs with no improvement, altering masks... ')
        self._improve_masks(self._train_images, self._train_masks)
        self._improve_masks(self._val_imgs, self._val_masks)

        self._epochs_from_last_model_improvement = 0
        self._patience += self._patience_increase
        self._mask_weaken_modifier += self._mask_weaken_modifier_decay
        if self._mask_weaken_modifier > 1.0:
            self._mask_weaken_modifier = 1.0
        print(f'New patience: {self._patience}, new mask weaken modifier: {self._mask_weaken_modifier}.')

        self._net.model.load_weights(self.model_initial_weights_path)

    def _improve_masks(self, imgs, masks):
        size_h, size_w = self._network_input_wh
        for i, (img, mask) in enumerate(zip(imgs, masks)):
            step_h, step_w = (img.shape[0] - size_h) // self._eval_step_ratio, (
                        img.shape[1] - size_w) // self._eval_step_ratio
            probabs = np.zeros(img.shape[:2], dtype=np.float32)
            probabs_overlap_counter = np.zeros(img.shape[:2], dtype=np.float32)

            img_out = np.zeros([*img.shape[:2], 6], dtype=np.float32)

            img_lab = color.rgb2lab(img)

            img = img.astype(np.float32)
            img = norm_img(img)

            img_lab[:, :, 0] = img_lab[:, :, 0] / 100.0
            img_lab[:, :, 1] = (img_lab[:, :, 1] + 127.0) / 255.0
            img_lab[:, :, 2] = (img_lab[:, :, 2] + 127.0) / 255.0

            img_out[:, :, :3] = img
            img_out[:, :, 3:] = img_lab

            patches = []
            patches_coords = []
            h = 0
            while h + size_h <= img.shape[0]:
                w = 0
                while w + size_w <= img.shape[1]:
                    patch = img_out[h:h + size_h, w:w + size_w, :]
                    #patch.shape = [1, *patch.shape]
                    patches.append(patch)
                    patches_coords.append([h, w])

                    w += step_w
                    if step_w == 0:
                        break
                h += step_h
                if step_h == 0:
                    break

            patches = np.array(patches)
            result = self._model.predict(patches, verbose=0, batch_size=32)[:, :, :, 0]
            for j, coords in enumerate(patches_coords):
                h, w = coords
                probabs[h:h + size_h, w:w + size_w] += result[j]
                probabs_overlap_counter[h:h + size_h, w:w + size_w] += 1.0

            probabs_overlap_counter[probabs_overlap_counter == 0.0] = 1.0
            probabs /= probabs_overlap_counter

            fmask = mask / 255.0
            new_mask = fmask + probabs[..., np.newaxis] / 2.0
            new_mask = ((-200**-new_mask)+1)*3-2
            new_mask[new_mask < 0.0] = 0.0
            masks[i] = np.array(new_mask * 255, dtype=mask.dtype)
            pass
