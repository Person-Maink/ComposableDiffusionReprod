import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from torchvision.datasets import FashionMNIST
from torchvision.utils import make_grid, save_image
from torch_ema import ExponentialMovingAverage as EMA
from tqdm import tqdm
from dataloader import ShapesDatasetOneHot, VEC_LEN, visualize_batch
from random import sample

from smalldiffusion import (
    ScheduleLogLinear, samples, training_loop, MappedDataset, Unet, Scaled,
    img_train_transform, img_normalize
)

MAX_OBJECTS = 5
COND_DIM    = VEC_LEN * MAX_OBJECTS   # 16 × 5 = 80
TEMB_CH = 64 * 4

import os
print(os.getcwd())



def main(train_batch_size=24, epochs=300, sample_batch_size=64, test=True):
    if test: 
        a = Accelerator()
        print(a.device)  

        cond_embed = torch.nn.Linear(COND_DIM, TEMB_CH)
        model = Scaled(Unet)(64, 3, 3,
                            ch=64, ch_mult=(1, 1, 2), attn_resolutions=(14,),
                            cond_embed=cond_embed)
        model.load_state_dict(torch.load('trained/checkpoint_125000.pth'))
        
        schedule = ScheduleLogLinear(sigma_min=0.01, sigma_max=20, N=800)
        
        test_ds = ShapesDatasetOneHot('dataset/shape_dataset_test', 'dataset/metadata_test.json',
                              img_transform=img_train_transform)
        test_loader = DataLoader(
            MappedDataset(test_ds, lambda x: (x[0], x[1].flatten(0))),
            batch_size=64, shuffle=False, pin_memory=True
        )

        model.eval()
        model, test_loader = a.prepare(model, test_loader)

        for i, (image, conds) in tqdm(enumerate(test_loader)):
            image = image.to(a.device)
            conds = torch.stack([c.flatten(0) for c in conds]).to(a.device)

            sigmas = schedule.sample_sigmas(20)
            *_, x0_pred = samples(model, sigmas, gam=1.6,
                                batchsize=64, cond=conds, accelerator=a)

            save_image(img_normalize(make_grid(x0_pred-image)), f'samples_test_{i}.png')

    else:
    # Setup
        a = Accelerator()
        print(a.device)

        raw_ds = ShapesDatasetOneHot('dataset/shape_dataset', 'dataset/metadata.json',
            img_transform=img_train_transform)

        loader = DataLoader(
            MappedDataset(raw_ds, lambda x: (x[0], x[1].flatten(0))),
            batch_size=train_batch_size, shuffle=True, pin_memory=True,
        )

        test_ds = ShapesDatasetOneHot('dataset/shape_dataset_test', 'dataset/metadata_test.json',
                              img_transform=img_train_transform)
        test_loader = DataLoader(
            MappedDataset(test_ds, lambda x: (x[0], x[1].flatten(0))),
            batch_size=10, shuffle=False, pin_memory=True
        )

        schedule = ScheduleLogLinear(sigma_min=0.01, sigma_max=20, N=800)

        cond_embed = torch.nn.Linear(COND_DIM, TEMB_CH)
        model = Scaled(Unet)(64, 3, 3,
                            ch=64, ch_mult=(1, 1, 2), attn_resolutions=(14,),
                            cond_embed=cond_embed)

        # Train
        ema = EMA(model.parameters(), decay=0.999)
        ema.to(a.device)
        COUNTER = 0 
        for ns in training_loop(loader, model, schedule, epochs=epochs, lr=5e-6, accelerator=a,
                                conditional=True):
            ns.pbar.set_description(f'Loss={ns.loss.item():.5}')
            ema.update()
            
            if COUNTER % 1_000 == 0:
                with ema.average_parameters():
                    model.eval()
                    rand_indices = sample(range(len(test_ds)), 10)
                    images, conds = zip(*[test_ds[i] for i in rand_indices])
                    images = torch.stack(images).to(a.device)
                    conds = torch.stack([c.flatten(0) for c in conds]).to(a.device)

                    sigmas = schedule.sample_sigmas(20)
                    *_, x0_pred = samples(model, sigmas, gam=1.6,
                                        batchsize=10, cond=conds, accelerator=a)

                    save_image(img_normalize(make_grid(x0_pred)), f'samples_{COUNTER}.png')
                    # visualize_batch((x0_pred, conds.reshape(10, 5, -1)))
                    model.train()
                torch.save(model.state_dict(), f'checkpoint_{COUNTER}.pth')

            COUNTER += 1
        # Sample
        with ema.average_parameters():
            dummy_cond = torch.zeros(sample_batch_size, COND_DIM, device=a.device)

            *_, x0 = samples(model, schedule.sample_sigmas(20), gam=1.6,
                            batchsize=sample_batch_size, cond=dummy_cond, accelerator=a)

            save_image(img_normalize(make_grid(x0)), 'samples.png')
            torch.save(model.state_dict(), 'checkpoint.pth')

if __name__=='__main__':
    main()
