import open_clip                                           # pip install open-clip-torch
import matplotlib.pyplot as plt
import torch
import json, pathlib, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor



def visualize_batch(batch, *, n_cols: int = 8) -> None:
    imgs, emb = batch
    imgs = imgs.detach().cpu()
    emb  = emb.detach().cpu().float()
    emb = emb.permute(0, 2, 1).reshape(emb.shape[0], -1)  # (B, M×16)

    B = imgs.size(0)
    n_rows = (B + n_cols - 1) // n_cols
    fig_imgs, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    axes = axes.flatten()
    for ax in axes:
        ax.axis('off')

    for i in range(B):
        img = imgs[i]
        if img.min() < 0:                         # de-normalize if necessary
            img = (img * 0.5) + 0.5
        axes[i].imshow(img.permute(1, 2, 0).clamp(0, 1))

    plt.tight_layout()
    # plt.imshow(emb2d, aspect='auto')
    plt.show()

    fig_emb, ax_emb = plt.subplots(figsize=(10, 4))
    im = ax_emb.imshow(emb, aspect='auto', interpolation='nearest')
    ax_emb.set_xlabel('Embedding dimension')
    ax_emb.set_ylabel('Batch index')
    fig_emb.colorbar(im, ax=ax_emb, fraction=0.025)
    plt.tight_layout()
    plt.show()


SHAPES = ['circle', 'triangle', 'square', 'diamond', 'star', 'hexagon']  # 6
COLORS = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'black', 'white']  # 8
VEC_LEN = len(SHAPES) + len(COLORS) + 2                                  # 16 per object

class ShapesDatasetOneHot(Dataset):
    def __init__(self, img_root: str, meta_file: str,
                 img_transform=None, max_objects: int = 5):
        self.img_root   = pathlib.Path(img_root)
        self.transform  = img_transform or ToTensor()
        self.max_obj    = max_objects

        with open(meta_file) as f:
            self.meta = json.load(f)
        self.fnames = sorted(self.meta.keys())
        self.img_size =  self._get_img_size()

    def _get_img_size(self):
        fn = self.img_root / self.fnames[0]
        return Image.open(fn).size[0]       # assume square images

    # ----- helper -------------------------------------------------------
    def _encode_one(self, shape, colour, xy):
        s_vec = torch.nn.functional.one_hot(
                    torch.tensor(SHAPES.index(shape)), len(SHAPES)).float()
        c_vec = torch.nn.functional.one_hot(
                    torch.tensor(COLORS.index(colour)), len(COLORS)).float()
        x, y  = torch.tensor(xy, dtype=torch.float32) / self.img_size
        return torch.cat([s_vec, c_vec, torch.tensor([x, y])])

    # ----- dataset ------------------------------------------------------
    def __len__(self): return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        rec   = self.meta[fname]
        img   = self.transform(Image.open(self.img_root / fname).convert('RGB'))

        vecs = [self._encode_one(s, c, xy)
                for s, xy, c in zip(rec['shapes'], rec['coordinates'], rec['colors'])]
        pad  = self.max_obj - len(vecs)
        if pad < 0:
            vecs = vecs[:self.max_obj]      # truncate if too many objects
        else:
            vecs.extend([torch.zeros(VEC_LEN)] * pad)

        cond = torch.stack(vecs)            # (max_objects, 16)
        return img, cond.float()


if __name__ == '__main__':
    ds  = ShapesDatasetOneHot('dataset/shape_dataset', 'dataset/metadata.json',
                              max_objects=5)
    ldr = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4,
                     pin_memory=True)
    batch = next(iter(ldr))   # imgs: (B,3,H,W)   conds: (B,M,16)
    visualize_batch(batch)

