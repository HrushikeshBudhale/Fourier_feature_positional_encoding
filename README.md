# Fourier_feature_positional_encoding
Experiments with 2D positional encoding

minimal implementation of paper [Fourier Features Let Networks Learn
High Frequency Functions in Low Dimensional Domains](https://arxiv.org/pdf/2006.10739.pdf)

## Input

|          Input           |          Learned (L=256,M=10)             |
| :-----------------------: | :----------------------: |
| ![origin](data/dryfruit.jpg "Origin") | ![recon](output/dryfruit_L256_m10.gif "Recon") |

## Experiments
For positional encoding we can experiment with number of frequencies and multiplier to choose the best values for highest psnr.
L defining number of frequencies and M being frequency multiplier. Best result was achieved using L=256 and M=10

|![](output/dryfruit_L16_m10.gif)|![](output/dryfruit_L64_m10.gif)|
|:-:|:-:|
|With L=16 & M=10|With L=64 & M=10|


|![](output/dryfruit_L256_m1.gif)|![](output/dryfruit_L256_m50.gif)|
|:-:|:-:|
|With L=256 & M=1|With L=256 & M=50|
