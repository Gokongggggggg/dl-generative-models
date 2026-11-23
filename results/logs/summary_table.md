# Final Test-Set Metrics (7,000 images)

| model              |   latent_dim |   train_time_per_epoch_s |   latency_ms_batch64 | test_mse   | test_psnr_db   | test_ssim   | fid      |
|:-------------------|-------------:|-------------------------:|---------------------:|:-----------|:---------------|:------------|:---------|
| Vanilla AE         |           64 |                   3.8028 |               0.9539 | 0.0069     | 21.6007        | 0.8208      | N/A      |
| Sparse AE          |           64 |                   3.2029 |               1.193  | 0.0074     | 21.3258        | 0.8108      | N/A      |
| VAE                |           64 |                   4.0949 |               1.2003 | 0.0118     | 19.2783        | 0.7271      | 79.4156  |
| Vanilla GAN        |          100 |                   7.2601 |               0.8211 | N/A        | N/A            | N/A         | 120.0865 |
| DCGAN              |          100 |                  15.0407 |               4.3753 | N/A        | N/A            | N/A         | 11.5267  |
| DDPM               |          784 |                  34.1979 |            2824.72   | N/A        | N/A            | N/A         | 39.2816  |
| Frozen MLP Decoder |           64 |                   3.0586 |               1.0031 | 0.0155     | 18.1095        | 0.686       | N/A      |

N/A: reconstruction metrics for GAN/DCGAN/DDPM (no encoder); FID for Vanilla/Sparse AE (not generative priors).
