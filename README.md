# DPMS
> `A Strong Baseline for Semi-supervised Medical Image Segmentation`
> 
> Efforts in SSL research should prioritize simpler designs.
>
> I initialliy aim to submit this work to IPMI 2023 ... However, rather than letting it lie dormant on Overleaf, I've decided to share this simple work to inspire more interesting SSL studies.


My studies on Semi-supervised segmentation:
- **iMAS** [(CVPR'23-Code)](https://github.com/ZhenZHAO/iMAS): reserve the strong augmentations, but apply them based on the model training status and the instance differences. i.e., instance-specific and Model-adaptive Supervision
- **Augseg** [(CVPR'23-Code)](https://github.com/ZhenZHAO/AugSeg): revise the strong augmentation to prevent over-perturbations, increase the randomness (continuous v.s. discrete) to produce more prediction disagreement.
- **DPMS**: reserve the strong augmentation, but apply simple yet effective stablization strategies. Simplest but equally effective.

## Performance on 2D and 3D datasets

On 2D acdc using Unet as backbone

| Methods         | Labels on 2D ACDC | Dice($\uparrow$)     | Jaccard($\uparrow$)  | 95HD($\downarrow$)    | ASD($\downarrow$)     |
| --------------- | :---------------: | -------------------- | -------------------- | ------------------- | ------------------- |
| UA-MT           |   3 labels (5%)   | 46.04                | 35.97                | 20.08               | 7.75                |
| SASSNet         |         -         | 57.77                | 46.14                | 20.05               | 6.06                |
| DTC             |         -         | 56.90                | 45.67                | 23.36               | 7.39                |
| URPC            |         -         | 55.87                | 44.64                | 13.60               | 3.74                |
| MC-Net          |         -         | 62.85                | 52.29                | 7.62                | 2.33                |
| SS-Net          |         -         | 65.82                | 55.38                | 6.67                | 2.28                |
| CT-CT           |         -         | 65.50                | -                    | 16.2                | -                   |
| **DPMS (Ours)** |         -         | **88.44** $\pm 0.15$ | **80.00** $\pm 0.20$ | **2.03** $\pm 0.56$ | **0.59** $\pm 0.09$ |




On 3D LA using Vnet as backbone

| Methods         | Labels on 3D LA | Dice($\uparrow$) | Jaccard($\uparrow$) | 95HD($\downarrow$) | ASD($\downarrow$) |
| --------------- | :-------------: | ---------------- | ------------------- | ---------------- | --------------- |
| UA-MT           |    4 labels (5%)    | 82.26            | 70.98               | 13.71            | 3.82            |
| SASSNet         |        -        | 81.60            | 69.63               | 16.16            | 3.58            |
| DTC             |        -        | 81.25            | 69.33               | 14.90            | 3.99            |
| URPC            |        -        | 82.48            | 71.35               | 14.65            | 3.65            |
| MC-Net          |        -        | 83.59            | 72.36               | 14.07            | 2.70            |
| SS-Net          |        -        | 86.33            | 76.15               | 9.97             | 2.31            |
| **DPMS (Ours)** |                 | **89.64**  $\pm 0.22$      |     **81.29** $\pm 0.35$            |        **5.99** $\pm 0.40$      | **1.77** $\pm 0.05$ |


On 3D Pancreas using Vnet as backbone

| Methods         | Labels on 3D LA | Dice($\uparrow$) | Jaccard($\uparrow$) | 95HD($\downarrow$) | ASD($\downarrow$) |
| --------------- | :-------------: | ---------------- | ------------------- | ---------------- | --------------- |
| UA-MT           |    6 labels (10%)    | 66.44            | 52.02              | 17.04            | 3.03            |
| SASSNet         |        -        | 68.97            | 54.29               | 18.83            | 1.96            |
| DTC             |        -        | 66.58            | 51.79               | 15.46            | 4.16            |
| URPC            |        -        | 73.53            | 59.44               | 22.57            | 7.85            |
| MC-Net          |        -        | 69.07            | 54.36               | 14.53            | 2.28            |
| **DPMS (Ours)** |                 | **79.88**  $\pm 0.56$      |     **67.14** $\pm 0.48$            |        **7.77** $\pm 1.45$      | **1.56** $\pm 0.26$ |



