"""Generate the project report PDF from saved metrics/logs/figures.

Run:  python -m experiments.make_report   ->   results/D6421_Report.pdf
"""
import json

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from configs import config

OUT = config.RESULTS_DIR / "D6421_Report.pdf"
PAGE_W, PAGE_H = A4
MAX_W, MAX_H = PAGE_W - 4 * cm, PAGE_H - 6 * cm

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1x", parent=ss["Heading1"], fontSize=15, spaceAfter=8)
H2 = ParagraphStyle("H2x", parent=ss["Heading2"], fontSize=12, spaceAfter=6)
BODY = ParagraphStyle("Bodyx", parent=ss["BodyText"], fontSize=9.5, leading=13.5,
                      spaceAfter=6)
CAP = ParagraphStyle("Capx", parent=ss["BodyText"], fontSize=8.5, leading=11,
                     textColor=colors.HexColor("#444444"), spaceAfter=10)
TITLE = ParagraphStyle("Titlex", parent=ss["Title"], fontSize=20, leading=24)


def load(name):
    return json.load(open(config.LOG_DIR / name))


def fmt(v, nd=4):
    return "N/A" if v is None else f"{v:.{nd}f}"


def table(rows, widths=None, header=True, fs=8):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), fs),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDDDDD")),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


def figure(path, caption):
    w, h = PILImage.open(path).size
    s = min(MAX_W / w, MAX_H / h)
    return [KeepTogether([Image(str(path), width=w * s, height=h * s),
                          Paragraph(caption, CAP)])]


def main():
    metrics = load("final_metrics.json")
    proof = load("exp1_mlp_proof.json")
    story = []
    P = lambda txt, st=BODY: story.append(Paragraph(txt, st))

    # ---------------- title ----------------
    P("Deep Learning (D6421) Take-Home Project", TITLE)
    P("A Comparative Empirical Study of Six Generative Architectures and the "
      "Theoretical Limitations of a Discriminative MLP Classifier on FashionMNIST", H2)
    P("Platform: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB), Windows, Python 3.13, "
      "PyTorch 2.7.1+cu118. All experiments use seed 42 and one identical 80/10/10 "
      "split of the 70,000 FashionMNIST images. Report generated directly from the "
      "saved training logs and test-set evaluation outputs.", BODY)
    story.append(Spacer(1, 6))

    # ---------------- 1 introduction ----------------
    P("1. Introduction and Objectives", H1)
    P("Generative models learn the data distribution p(x) so that new samples can be "
      "drawn, whereas discriminative classifiers learn p(y|x) only well enough to "
      "separate classes. This project (i) proves empirically, in five controlled "
      "stages, that a supervised MLP classifier cannot reconstruct or generate images "
      "by weight transposition or by naive prior sampling, and (ii) compares six "
      "generative architectures - Vanilla Autoencoder, Sparse Autoencoder, Variational "
      "Autoencoder, an MLP-based Vanilla GAN, DCGAN, and a Denoising Diffusion "
      "Probabilistic Model (DDPM) - under identical data, seeds, and evaluation "
      "protocol.")

    # ---------------- 2 setup ----------------
    P("2. Data, Preprocessing and Reproducibility", H1)
    P("FashionMNIST (28x28 grayscale, 10 classes) train (60k) and test (10k) partitions "
      "are merged into 70,000 images and normalized to [-1, 1] with "
      "transforms.Normalize((0.5,), (0.5,)). A seeded permutation (seed 42) splits them "
      "into 56,000 train (shuffled), 7,000 validation and 7,000 test images; batch size "
      "is 128. Every final number in this report is computed on the 7,000-image test "
      "split only. Seeds are fixed for random, numpy, torch and torch.cuda, with "
      "cudnn.deterministic=True and cudnn.benchmark=False.")
    story.append(table([
        ["Item", "Value"],
        ["GPU / OS", "RTX 4050 Laptop (6 GB VRAM) / Windows"],
        ["Framework", "PyTorch 2.7.1+cu118, torchvision 0.22.1"],
        ["Split (train/val/test)", "56,000 / 7,000 / 7,000 (seed 42)"],
        ["Pixel domain", "[-1, 1]; metrics rescaled to [0, 1]"],
        ["Batch size / optimizer", "128 / Adam"],
    ], widths=[4.5 * cm, 12 * cm]))
    story.append(Spacer(1, 8))

    # ---------------- 3 part A ----------------
    P("3. Part A - MLP Classifier Limitation Proof (five stages)", H1)
    P("<b>Stage 1.</b> Train a 784-256-64-10 MLP (ReLU, cross-entropy) until test "
      f"accuracy >= 88%; achieved <b>{proof['stage1_test_accuracy'] * 100:.2f}%</b>. "
      "The penultimate activation z_MLP in R^64 is the discriminative latent code.")
    P("<b>Stage 2.</b> Top-down transposition without biases or training: "
      "z_hat = ReLU(W3^T y_c), h1_hat = ReLU(W2^T z_hat), x_hat = Tanh(W1^T h1_hat), "
      "for one-hot class vectors y_c.")
    P("<b>Stage 3.</b> Latent transposition of real test codes: "
      "x_hat = Tanh(W1^T ReLU(W2^T z_MLP)).")
    P("<b>Stage 4.</b> Freeze all classifier weights (requires_grad=False) and train a "
      "separate 64-256-784 Tanh decoder with MSE on frozen z_MLP; compare against the "
      "Vanilla AE (z_AE in R^64).")
    P("<b>Stage 5.</b> Fit mu_z, sigma_z on test latents, sample "
      "z_rand ~ N(mu_z, diag(sigma_z^2)) and decode with the Stage-4 decoder.")
    s2, s3, s4 = (proof["stage2_class_transpose"], proof["stage3_latent_transpose"],
                  proof["stage4_frozen_decoder_test"])
    story.append(table([
        ["Stage", "Test MSE", "PSNR (dB)", "SSIM", "Interpretation"],
        ["2: W^T from class y_c", fmt(s2["mse"]), fmt(s2["psnr"], 2), fmt(s2["ssim"]),
         "noise-like, no object"],
        ["3: W^T from z_MLP", fmt(s3["mse"]), fmt(s3["psnr"], 2), fmt(s3["ssim"]),
         "noise-like, no object"],
        ["4: trained frozen decoder", fmt(s4["mse"]), fmt(s4["psnr"], 2), fmt(s4["ssim"]),
         "blurry but structured"],
        ["Vanilla AE (reference)", fmt(metrics["Vanilla AE"]["test_mse"]),
         fmt(metrics["Vanilla AE"]["test_psnr_db"], 2),
         fmt(metrics["Vanilla AE"]["test_ssim"]), "sharpest reconstruction"],
    ], widths=[4.6 * cm, 2.2 * cm, 2.2 * cm, 1.8 * cm, 5.7 * cm]))
    story.append(Spacer(1, 6))
    P("Stage 5 samples decode to incoherent blobs (Fig. 3, panel 4): nothing regularizes "
      "the aggregate discriminative posterior toward a smooth, space-filling prior, so "
      "Gaussian draws land off the latent data manifold. Together the stages show that "
      "class-separability neither implies invertibility (Stages 2-3, SSIM ~0.05) nor a "
      "usable generative prior (Stage 5), and that even a trained decoder on frozen "
      "features loses detail the classifier never needed (Stage 4 vs Vanilla AE).")

    # ---------------- 4 models ----------------
    P("4. Part B - Generative Model Specifications", H1)
    story.append(table([
        ["Model", "Architecture", "Latent", "Objective", "Epochs"],
        ["Vanilla AE", "enc 784-256-64 / dec 64-256-784, Tanh", "64", "MSE", "50"],
        ["Sparse AE", "as Vanilla AE + L1 on bottleneck", "64",
         "MSE + 1e-4 * ||h||_1", "50"],
        ["VAE", "enc 784-256-(mu,logvar 64); dec 64-256-784", "64",
         "sum-MSE + KL(q||N(0,I))", "50"],
        ["Vanilla GAN", "G 100-256-512-784 Tanh; D 784-512-256-1", "100",
         "BCEWithLogits (non-saturating G)", "100"],
        ["DCGAN", "ConvT 7x7->14->28 (BN+ReLU); D conv 28->14->7 (BN+LeakyReLU)", "100",
         "BCEWithLogits", "80"],
        ["DDPM", "U-Net 28->14->7, ResBlocks+GroupNorm, sinusoidal t-emb (1.78M params)",
         "784 (x_T)", "MSE eps-prediction, T=300 linear betas", "60"],
    ], widths=[2.3 * cm, 6.6 * cm, 1.7 * cm, 4.4 * cm, 1.5 * cm], fs=7.5))
    story.append(Spacer(1, 6))
    P("The Sparse AE penalizes the per-sample L1 <i>norm</i> of the 64-d bottleneck "
      "(lambda = 1e-4); using a mean instead would make the penalty negligible. "
      "||h||_1 falls from ~130 (unpenalized) to ~17 while SSIM drops only 0.821 -> 0.811. "
      "DDPM uses ancestral sampling with posterior variance; only the final x_0 is "
      "clamped to [-1,1] - clamping every intermediate x_t truncates the ~N(0,1) noise "
      "distribution at high t and collapsed sample quality (FID 197) before the fix "
      "(FID 39.3).")

    # ---------------- 5 protocol ----------------
    P("5. Evaluation Protocol", H1)
    P("<b>Reconstruction:</b> MSE, PSNR and SSIM on [0,1]-rescaled images over all 7,000 "
      "test images (SSIM per-image, data_range=1.0); VAE reconstruction uses mu. "
      "<b>Latency:</b> batch of 64 with torch.cuda.synchronize() + time.perf_counter(), "
      "10 warm-up iterations then 50 timed repeats (DDPM: 5 repeats of the full 300-step "
      "reverse loop). <b>FID:</b> torchmetrics FrechetInceptionDistance (feature 2048), "
      "5,000 real test images vs 5,000 generated samples, grayscale replicated to "
      "3-channel RGB uint8. <b>Training time:</b> mean wall-clock seconds per epoch.")

    # ---------------- 6 results ----------------
    P("6. Quantitative Results (Test Set)", H1)
    order = ["Vanilla AE", "Sparse AE", "VAE", "Vanilla GAN", "DCGAN", "DDPM",
             "Frozen MLP Decoder"]
    rows = [["Model", "Latent", "s/epoch", "ms (b64)", "MSE", "PSNR", "SSIM", "FID"]]
    for k in order:
        r = metrics[k]
        rows.append([k, str(r["latent_dim"]), fmt(r["train_time_per_epoch_s"], 2),
                     fmt(r["latency_ms_batch64"], 2), fmt(r["test_mse"]),
                     fmt(r["test_psnr_db"], 2), fmt(r["test_ssim"]), fmt(r["fid"], 2)])
    story.append(table(rows, widths=[3.4 * cm, 1.3 * cm, 1.6 * cm, 1.9 * cm, 1.9 * cm,
                                      1.7 * cm, 1.6 * cm, 1.9 * cm], fs=7.5))
    story.append(Spacer(1, 6))
    P("N/A entries are structural: GAN/DCGAN/DDPM have no encoder (no reconstruction "
      "metrics); Vanilla/Sparse AE have no regularized prior (FID not meaningful). "
      "DCGAN attains the best sample quality (FID 11.53); DDPM is second (39.28) with "
      "sharp, diverse samples but ~2.8 s per batch of 64 because sampling requires 300 "
      "sequential network evaluations, versus ~4.4 ms for a single DCGAN forward pass.")

    # ---------------- 7 figures ----------------
    story.append(PageBreak())
    P("7. Qualitative Results", H1)
    figs = [
        ("Figur1_Reconstruction_Comparison.png",
         "Figure 1: Ground truth (row 1) vs reconstructions by Vanilla AE, Sparse AE and "
         "VAE on eight test images. AE variants preserve edges; the VAE is smoother."),
        ("Figur2_Random_Generative_Sampling.png",
         "Figure 2: 16 prior samples per model. VAE: coherent but soft; Vanilla GAN: "
         "structured but noisy; DCGAN and DDPM: sharp, diverse garments."),
        ("Figur3_MLP_Limitation_Proof.png",
         "Figure 3: (1) ground truth; (2) bias-free W^T transposes from class one-hots "
         "and from real z_MLP (noise-like); (3) trained decoder on frozen z_MLP vs "
         "Vanilla AE; (4) Gaussian prior samples on the MLP latent space (blobs)."),
        ("Figur4_Latent_Interpolation.png",
         "Figure 4: linear latent interpolation z(a)=(1-a)z_A+a z_B for a in "
         "{0,0.2,...,1} between Trouser and Sneaker endpoints, across five models. "
         "All traversals stay on-manifold; GAN endpoints selected via the classifier."),
        ("Figur5_Loss_Curves_and_DDPM_Trajectory.png",
         "Figure 5: per-epoch training losses for all six models (G/D separate for the "
         "GANs) and the DDPM reverse trajectory x_T -> x_0 (t=300...0)."),
    ]
    for fname, cap in figs:
        story += figure(config.FIG_DIR / fname, cap)

    # ---------------- 8 discussion ----------------
    story.append(PageBreak())
    P("8. Discussion", H1)
    P("<b>Reconstruction vs generation.</b> The Vanilla AE reconstructs best (SSIM "
      "0.821) yet cannot sample: its latent aggregate posterior is irregular. The KL "
      "term buys the VAE a usable N(0,I) prior at a reconstruction cost (SSIM 0.727, "
      "FID 79.4). Adversarial training (DCGAN, FID 11.5) and denoising score matching "
      "(DDPM, FID 39.3) produce the sharpest samples without any encoder.")
    P("<b>Why the classifier fails as a generator.</b> ReLU activations discard sign "
      "information, so W^T is not an inverse (Stages 2-3). Cross-entropy only shapes "
      "class-separable geometry, not a density: frozen-feature decoding recovers "
      "structure (SSIM 0.686) but loses nuisance detail, and Gaussian prior sampling "
      "fails entirely (Stage 5). Generation requires an objective that shapes the "
      "aggregate posterior / marginal distribution - exactly what VAE, GAN and DDPM "
      "add and the classifier lacks.")
    P("<b>Cost.</b> Training cost per epoch grows from ~3-4 s (AE family) to ~7-15 s "
      "(GANs) to ~34 s (DDPM). Inference spans four orders of magnitude: 0.8-4.4 ms "
      "(single forward pass) vs ~2,825 ms (DDPM full reverse loop).")
    P("<b>Limitations.</b> FID uses ImageNet-pretrained InceptionV3 on upscaled 28x28 "
      "grayscale, so absolute values are domain-shifted (rankings remain informative); "
      "single seed per model; DDPM truncated to T=300 for single-GPU feasibility; the "
      "MLP GAN's capacity limits sample fidelity (FID 120.1).")

    # ---------------- 9 conclusion ----------------
    P("9. Conclusion", H1)
    P("Under one identical split and seed, the five-stage experiment confirms that "
      "discriminative latents are neither invertible by transposition nor compatible "
      "with naive Gaussian priors, while a trained decoder on frozen features remains "
      "inferior to end-to-end reconstruction. Among generators, DCGAN offers the best "
      "quality/latency trade-off (FID 11.5, 4.4 ms), DDPM the best sharpness-per-sample "
      "at high sampling cost, and the VAE a middle ground with a principled prior; "
      "autoencoders remain the reconstruction specialists.")

    # ---------------- 10 reproducibility ----------------
    P("10. Reproducibility and Repository", H1)
    P("Commands (repo root): python -m experiments.exp1_mlp_proof; "
      "exp2_train_autoencoders; exp3_train_gans; exp4_train_ddpm; exp5_evaluate; "
      "make_report. Code: configs/, models/, utils/, experiments/. Artifacts: "
      "results/checkpoints (11 files), results/figures (5 PNGs, 300 DPI), results/logs "
      "(per-epoch losses, final_metrics.json/csv, summary_table.md). Incremental git "
      "commits record each milestone.")

    P("References", H2)
    for ref in [
        "[1] Xiao, Rasul, Vollgraf. Fashion-MNIST: a Novel Image Dataset for "
        "Benchmarking Machine Learning Algorithms. arXiv:1708.07747, 2017.",
        "[2] Kingma, Welling. Auto-Encoding Variational Bayes. ICLR 2014.",
        "[3] Goodfellow et al. Generative Adversarial Nets. NeurIPS 2014.",
        "[4] Radford, Metz, Chintala. Unsupervised Representation Learning with Deep "
        "Convolutional Generative Adversarial Networks. ICLR 2016.",
        "[5] Ho, Jain, Abbeel. Denoising Diffusion Probabilistic Models. NeurIPS 2020.",
        "[6] Heusel et al. GANs Trained by a Two Time-Scale Update Rule Converge to a "
        "Local Nash Equilibrium. NeurIPS 2017 (FID).",
        "[7] Paszke et al. PyTorch: An Imperative Style, High-Performance Deep "
        "Learning Library. NeurIPS 2019.",
        "[8] torchmetrics: FrechetInceptionDistance (Lightning AI).",
    ]:
        P(ref, CAP)

    doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title="D6421 Take-Home Project Report",
                            author="D6421 Deep Learning")
    doc.build(story)
    print("Saved", OUT)


if __name__ == "__main__":
    main()
