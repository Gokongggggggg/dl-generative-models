"""Generate the final submission report PDF following pdf_structure.txt blueprint.

Run:  python -m experiments.make_report
Out:  results/TUGAS_DL_SESI07_2802467382_WISELY_JANSON_HALIM.pdf  (target 11 pages)
"""
import json
import tempfile
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, FrameBreak, Image,
                                KeepTogether, NextPageTemplate, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

from configs import config
from utils.diagrams import draw_architectures, render_equation

OUT = config.RESULTS_DIR / "TUGAS_DL_SESI07_2802467382_WISELY_JANSON_HALIM.pdf"
REPO_URL = "https://github.com/Gokongggggggg/dl-generative-models"
PAGE_W, PAGE_H = A4
MARGIN = 2 * cm
FRAME_W = PAGE_W - 2 * MARGIN
FRAME_H = PAGE_H - 2 * MARGIN
COL_W = (FRAME_W - 0.8 * cm) / 2

ss = getSampleStyleSheet()
TITLE = ParagraphStyle("T", parent=ss["Title"], fontSize=15, leading=19, spaceAfter=8)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=12.5, spaceAfter=6, spaceBefore=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=10.5, spaceAfter=4, spaceBefore=6)
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13.2, spaceAfter=5)
SMALL = ParagraphStyle("S", parent=ss["BodyText"], fontSize=8.5, leading=11.5, spaceAfter=4)
CAP = ParagraphStyle("C", parent=ss["BodyText"], fontSize=8.5, leading=11,
                     textColor=colors.HexColor("#333333"), spaceAfter=6)
EQCAP = ParagraphStyle("E", parent=ss["BodyText"], fontSize=8.5, leading=11, spaceAfter=7)


def load(name):
    return json.load(open(config.LOG_DIR / name))


def fmt(v, nd=4):
    return "N/A" if v is None else f"{v:.{nd}f}"


def table(rows, widths, fs=7.5):
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), fs),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#888888")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDDDDD")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return t


def img_flow(path, max_w, max_h=None):
    w, h = PILImage.open(path).size
    s = max_w / w
    if max_h:
        s = min(s, max_h / h)
    return Image(str(path), width=w * s, height=h * s)


def main():
    metrics = load("final_metrics.json")
    proof = load("exp1_mlp_proof.json")
    tmp = Path(tempfile.mkdtemp(prefix="d6421_eq_"))
    arch_png = config.FIG_DIR / "Arsitektur_7_Model.png"
    draw_architectures(arch_png)

    eqs = {
        "ae": r"$\mathcal{L}_{AE}=\frac{1}{N}\sum_{i=1}^{N}\|\,x_i-D_\phi(E_\theta(x_i))\,\|_2^2$",
        "sae": r"$\mathcal{L}_{SAE}=\mathcal{L}_{AE}+\lambda\sum_{j}|h_j|,\quad\lambda=10^{-4}$",
        "vae": r"$\mathcal{L}_{VAE}=-\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]+D_{KL}(q_\phi(z|x)\,\|\,\mathcal{N}(0,I))$",
        "reparam": r"$z=\mu+\sigma\odot\epsilon,\quad\epsilon\sim\mathcal{N}(0,I)$",
        "gan": r"$\min_G\max_D\,V(D,G)=\mathbb{E}_{x\sim p_{data}}[\log D(x)]+\mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]$",
        "ddpm": r"$\mathcal{L}_{simple}=\mathbb{E}_{t,x_0,\epsilon}\left[\|\,\epsilon-\epsilon_\theta(\sqrt{\bar{\alpha}_t}\,x_0+\sqrt{1-\bar{\alpha}_t}\,\epsilon,\ t)\,\|_2^2\right]$",
        "wt": r"$W^{(l)\,T}W^{(l)}\neq I\ \Rightarrow\ W^{(l)\,T}\neq (W^{(l)})^{-1}$",
        "relu": r"$\mathrm{ReLU}(u)=\max(0,u)$",
        "transpose": r"$\hat{x}=\tanh(W_1^T\,\mathrm{ReLU}(W_2^T\,\mathrm{ReLU}(W_3^T\,y_c)))$",
        "cond": r"$\hat{x}_{MSE}=\mathbb{E}[x\,|\,z]$",
        "js": r"$\min_G\,D_{JS}(P_{data}\,\|\,P_g),\quad D_{JS}=	frac{1}{2}D_{KL}(P_{data}\|M)+	frac{1}{2}D_{KL}(P_g\|M)$",
    }
    eq_png = {}
    for k, tex in eqs.items():
        p = tmp / f"eq_{k}.png"
        render_equation(tex, p)
        eq_png[k] = p

    story = []
    P = lambda txt, st=BODY: story.append(Paragraph(txt, st))
    EQ = lambda k, cap: story.append(KeepTogether(
        [img_flow(eq_png[k], FRAME_W - 1 * cm), Paragraph(cap, EQCAP)]))

    # ================= Halaman 1: identitas + abstrak =================
    P("Studi Komparatif Empiris Model Generatif (Vanilla AE, Sparse AE, VAE, Vanilla GAN, "
      "DCGAN, dan DDPM) serta Pembuktian Batasan Teoretis Model Klasifikasi untuk Image "
      "Generation pada Dataset Fashion-MNIST", TITLE)
    P("<b>Wisely Janson Halim</b> &nbsp;|&nbsp; NIM: 2802467382 &nbsp;|&nbsp; "
      "Deep Learning (D6421) - Take-Home Project", BODY)
    P(f'Tautan repositori (publik, aktif): <link href="{REPO_URL}" color="#0000CC">'
      f'{REPO_URL}</link>', BODY)
    P("<b>Spesifikasi hardware & lingkungan eksekusi.</b> Seluruh eksperimen dijalankan "
      "pada GPU lokal NVIDIA GeForce RTX 4050 Laptop (6 GB VRAM), Windows, Python 3.13.4, "
      "PyTorch 2.7.1+cu118, torchvision 0.22.1, torchmetrics 1.9.0. Deterministic seed "
      "<b>42</b> ditetapkan pada random, numpy, torch, dan torch.cuda dengan "
      "cudnn.deterministic=True serta cudnn.benchmark=False; shuffle DataLoader dan "
      "seluruh sampling memakai generator ber-seed.", BODY)
    story.append(table([
        ["Item", "Nilai"],
        ["Dataset", "Fashion-MNIST 28x28 grayscale, 10 kelas, 70.000 citra"],
        ["Split (seed 42)", "Train 56.000 (shuffle) / Val 7.000 / Test 7.000"],
        ["Normalisasi", "transforms.Normalize((0.5,), (0.5,)) -> [-1, 1]"],
        ["Batch / optimizer", "128 / Adam"],
        ["Evaluasi akhir", "ketat pada 7.000 citra Test Set"],
    ], widths=[4.2 * cm, 12.3 * cm]))
    story.append(Spacer(1, 6))
    P("Abstrak", H2)
    P("Studi ini membandingkan enam arsitektur generatif dan satu eksperimen batasan "
      "klasifier di bawah satu protokol terstandarisasi (split 80/10/10, seed 42, "
      "domain piksel [-1,1]). Fidelitas rekonstruksi terbaik dipegang Vanilla AE "
      "(SSIM 0,821; PSNR 21,60 dB), diikuti Sparse AE (0,811) dan VAE (0,727); kualitas "
      "sampel terbaik dipegang DCGAN (FID 11,53), lalu DDPM (39,28), VAE (79,42), dan "
      "Vanilla GAN (120,09), dengan trade-off latency ekstrem: 0,8-4,4 ms per batch 64 "
      "untuk model sekali forward versus 2.825 ms untuk loop reverse DDPM 300 langkah. "
      "Eksperimen lima tahap membuktikan klasifier MLP (akurasi test 88,44%) tidak dapat "
      "merekonstruksi maupun melakukan generasi: proyeksi balik W^T menghasilkan SSIM ~0,05, "
      "decoder pada fitur beku kalah dari AE (SSIM 0,686), dan sampling prior Gaussian "
      "pada ruang laten MLP menghasilkan blob tak koheren. Temuan menegaskan bahwa "
      "objective diskriminatif P(y|x) tidak membentuk prior P(z) yang dapat disampling.",
      BODY)

    # ================= Halaman 2-3: Bagian II =================
    story.append(PageBreak())
    P("Bagian II - Landasan Teori, Desain Arsitektur, dan Formulasi Matematika", H1)
    P("2.1 Diagram Skematik Arsitektur", H2)
    story.append(KeepTogether([img_flow(arch_png, FRAME_W),
                                 Paragraph("Gambar II.1: skema tujuh model (klasifier MLP + enam generatif). Semua encoder/"
      "decoder MLP memakai ReLU kecuali output Tanh; DCGAN memakai BatchNorm+ReLU "
      "(generator) dan BatchNorm+LeakyReLU(0.2) (discriminator); U-Net DDPM memakai "
      "ResBlock+GroupNorm dengan sinusoidal time embedding dan skip connection.", CAP)]))
    P("2.2 Tabel Konfigurasi Parameter Terstandarisasi", H2)
    story.append(table([
        ["Model", "Layer architecture", "Dim laten", "Aktivasi", "Optim", "LR", "Epoch"],
        ["MLP Classifier", "784-256-64-10", "64", "ReLU", "Adam", "1e-3", "10*"],
        ["Vanilla AE", "784-256-64 / 64-256-784", "64", "ReLU/Tanh", "Adam", "1e-3", "50"],
        ["Sparse AE", "sama + L1 bottleneck", "64", "ReLU/Tanh", "Adam", "1e-3", "50"],
        ["VAE", "784-256-(mu,logvar 64) / 64-256-784", "64", "ReLU/Tanh", "Adam", "1e-3", "50"],
        ["Vanilla GAN", "G 100-256-512-784; D 784-512-256-1", "100", "LReLU/Tanh", "Adam", "2e-4", "100"],
        ["DCGAN", "ConvT 7-14-28; D conv 28-14-7", "100", "BN+ReLU/LReLU", "Adam", "2e-4", "80"],
        ["DDPM", "U-Net 28-14-7, 1.78M params", "784", "SiLU/GroupNorm", "Adam", "2e-4", "60"],
    ], widths=[2.5 * cm, 5.6 * cm, 1.6 * cm, 2.5 * cm, 1.3 * cm, 1.3 * cm, 1.5 * cm], fs=7))
    P("*MLP dilatih hingga akurasi test >= 88% (tercapai epoch 10). Frozen decoder "
      "64-256-784 (Tanh, MSE) dilatih 50 epoch dengan seluruh bobot klasifier dibekukan. "
      "GAN memakai betas (0.5, 0.999); DCGAN init N(0, 0.02).", CAP)
    P("2.3 Formulasi Matematika Loss Function", H2)
    EQ("ae", "Vanilla AE: MSE rekonstruksi end-to-end antara piksel input dan output decoder.")
    EQ("sae", "Sparse AE: penalti L1 pada aktivasi bottleneck h (norm per-sampel); "
       "lambda = 1e-4 mendorong kode part-based yang sparse.")
    EQ("vae", "VAE (negative ELBO): suku rekonstruksi + KL divergence terhadap prior "
       "N(0,I); implementasi memakai sum-MSE per citra (decoder Gaussian).")
    EQ("reparam", "Reparameterization trick: gradien mengalir melalui mu dan sigma "
       "sementara stochasticity dipindah ke epsilon.")
    EQ("gan", "Vanilla GAN & DCGAN: minimax game; implementasi memakai BCEWithLogits "
       "dengan non-saturating generator loss (-log D(G(z))).")
    EQ("ddpm", "DDPM: objective prediksi noise (eps-prediction) dengan jadwal beta "
       "linear T=300 (1e-4 -> 0.02); alpha_bar_t = prod(1-beta_s).")

    # ================= Halaman 4-5: Bagian III =================
    story.append(PageBreak())
    P("Bagian III - Pembuktian Teoretis dan Empiris Batasan MLP Classifier", H1)
    P("3.1 Analisis Matematis Tiga Batasan Utama", H2)
    P("<b>(i) Information bottleneck diskriminatif.</b> Cross-entropy hanya mengoptimasi "
      "P(y|x); variasi spasial intra-kelas (pose, tekstur, tepi) tidak diperlukan untuk "
      "memisahkan kelas sehingga terbuang dari z_MLP. Laten diskriminatif cukup "
      "class-separable, tidak perlu invertible maupun space-filling.")
    P("<b>(ii) Non-invertibilitas aljabar dan aktivasi.</b> Matriks bobot persegi panjang "
      "dan non-ortogonal memenuhi:")
    EQ("wt", "sehingga transpos bukan invers: proyeksi balik W^T tidak memulihkan input.")
    EQ("relu", "ReLU memotong seluruh nilai negatif (informasi sign hilang), sehingga "
       "pemetaan forward many-to-one dan tidak dapat dibalik secara eksak.")
    EQ("transpose", "Rantai proyeksi balik Stage 2 (tanpa bias, tanpa training); Stage 3 "
       "memakai z_MLP riil menggantikan ReLU(W3^T y_c).")
    P("<b>(iii) Ketiadaan regularisasi prior P(z).</b> Tidak ada suku KL maupun "
      "adversarial yang membentuk aggregate posterior q(z) = E_x[q(z|x)] mendekati "
      "prior halus; terbentuk latent voids - daerah berkepadatan nol di antara modus "
      "kelas - sehingga sampling N(mu_z, diag(sigma_z^2)) jatuh di luar manifold data.")
    P("3.2 Hasil Empiris Lima Tahap & Komparasi Z_MLP vs Z_AE", H2)
    P(f"<b>Stage 1:</b> akurasi test klasifier <b>{proof['stage1_test_accuracy'] * 100:.2f}%</b> "
      "(>= 88% terpenuhi). <b>Stage 2-3:</b> proyeksi W^T dari one-hot y_c dan dari z_MLP "
      "riil. <b>Stage 4:</b> decoder 64-256-784 dilatih dengan MSE pada z_MLP beku "
      "(requires_grad=False). <b>Stage 5:</b> sampling z_rand ~ N(mu_z, diag(sigma_z^2)) "
      "lalu didecode decoder Stage 4.")
    s2, s3, s4 = (proof["stage2_class_transpose"], proof["stage3_latent_transpose"],
                  proof["stage4_frozen_decoder_test"])
    story.append(table([
        ["Konfigurasi (dim laten 64)", "Test MSE", "PSNR (dB)", "SSIM"],
        ["Stage 2: W^T dari y_c", fmt(s2["mse"]), fmt(s2["psnr"], 2), fmt(s2["ssim"])],
        ["Stage 3: W^T dari z_MLP", fmt(s3["mse"]), fmt(s3["psnr"], 2), fmt(s3["ssim"])],
        ["Stage 4: Frozen Z_MLP decoder", fmt(s4["mse"]), fmt(s4["psnr"], 2), fmt(s4["ssim"])],
        ["Vanilla AE (Z_AE, referensi)", fmt(metrics["Vanilla AE"]["test_mse"]),
         fmt(metrics["Vanilla AE"]["test_psnr_db"], 2), fmt(metrics["Vanilla AE"]["test_ssim"])],
    ], widths=[7.0 * cm, 3.0 * cm, 3.0 * cm, 3.5 * cm]))
    story.append(Spacer(1, 6))
    story.append(KeepTogether([img_flow(config.FIG_DIR / "Figur3_MLP_Limitation_Proof.png", FRAME_W),
                                 Paragraph("Figur 3: (1) ground truth; (2) transpose W^T dari y_c dan z_MLP - pola noise tanpa "
      "struktur objek; (3) decoder pada z_MLP beku (kabur namun terstruktur) vs Vanilla AE "
      "(tepi tajam); (4) sampling prior Gaussian pada laten MLP - blob tak koheren. "
      "Interpretasi: kegagalan Stage 2-3 mengonfirmasi batasan (ii); selisih Stage 4 vs AE "
      "mengonfirmasi (i) - detail yang tidak dibutuhkan untuk klasifikasi hilang permanen; "
      "kegagalan Stage 5 mengonfirmasi (iii).", CAP)]))

    # ================= Halaman 6: Bagian IV =================
    story.append(PageBreak())
    P("Bagian IV - Tabel Komparasi Metrik Numerik", H1)
    P("4.1 Protokol Evaluasi Test Set", H2)
    P("Seluruh angka dihitung pada 7.000 citra Test Set yang tidak pernah dilihat model "
      "selama training. MSE/PSNR/SSIM dihitung pada citra rescale [0,1] (SSIM per-citra, "
      "data_range=1.0; rekonstruksi VAE memakai mu). Inference latency: batch 64 dengan "
      "torch.cuda.synchronize() + time.perf_counter(), 10 iterasi warm-up lalu 50 pengulangan "
      "(DDPM: 5 pengulangan loop reverse penuh 300 langkah). FID: "
      "torchmetrics FrechetInceptionDistance (feature 2048), 5.000 citra test riil vs 5.000 "
      "sampel sintetis, grayscale direplikasi ke RGB uint8 [0,255]. Training time: rata-rata "
      "wall-clock detik per epoch dari log training.")
    P("4.2 Tabel Komparasi Kuantitatif Utama", H2)
    stab = {
        "Vanilla AE": "Stabil; MSE turun monoton",
        "Sparse AE": "Stabil; MSE + L1 konvergen",
        "VAE": "Stabil; recon & KL plateau",
        "Vanilla GAN": "Osilatif; G/D berfluktuasi",
        "DCGAN": "Cukup stabil; G naik di akhir",
        "DDPM": "Sangat stabil; MSE supervised",
    }
    rows = [["Arsitektur Model", "Dim laten", "Train s/epoch", "Latency ms/b64",
             "Test MSE", "PSNR (dB)", "SSIM", "FID", "Kestabilan Training"]]
    for k in ["Vanilla AE", "Sparse AE", "VAE", "Vanilla GAN", "DCGAN", "DDPM"]:
        r = metrics[k]
        rows.append([k, str(r["latent_dim"]), fmt(r["train_time_per_epoch_s"], 2),
                     fmt(r["latency_ms_batch64"], 2), fmt(r["test_mse"]),
                     fmt(r["test_psnr_db"], 2), fmt(r["test_ssim"]), fmt(r["fid"], 2),
                     stab[k]])
    story.append(table(rows, widths=[2.5 * cm, 1.2 * cm, 1.6 * cm, 1.8 * cm, 1.7 * cm,
                                     1.6 * cm, 1.4 * cm, 1.5 * cm, 3.6 * cm], fs=7))
    story.append(Spacer(1, 5))
    P("Alasan N/A: GAN/DCGAN/DDPM tidak memiliki encoder sehingga metrik rekonstruksi "
      "tidak terdefinisi; Vanilla/Sparse AE tidak memiliki prior teregularisasi sehingga "
      "FID sampling tidak bermakna. Baris tambahan Frozen MLP Decoder (dim 64; 3,06 s/epoch; "
      "1,00 ms; MSE 0,0155; PSNR 18,11; SSIM 0,686) dibahas di Bagian III.", CAP)

    # ================= Halaman 7-8: Bagian V (dua kolom) =================
    story.append(NextPageTemplate("two"))
    story.append(PageBreak())
    P("Bagian V - Kompilasi Visualisasi Komparatif", H1)

    def fig_unit(fname, cap, analysis, max_h=None):
        return KeepTogether([
            img_flow(config.FIG_DIR / fname, COL_W, max_h),
            Paragraph(cap, CAP), Paragraph(analysis, SMALL)])

    story.append(fig_unit(
        "Figur1_Reconstruction_Comparison.png",
        "Figur 1: grid rekonstruksi 8 citra test - Ground Truth vs Vanilla AE, Sparse AE, VAE.",
        "Vanilla AE dan Sparse AE mempertahankan tepi dan tekstur hampir identik dengan "
        "ground truth; VAE lebih halus karena suku KL mengorbankan detail high-frequency. "
        "Selisih SSIM 0,821/0,811 vs 0,727 terlihat konsisten sebagai blur ringan pada VAE."))
    story.append(FrameBreak())
    story.append(fig_unit(
        "Figur2_Random_Generative_Sampling.png",
        "Figur 2: 16 sampel prior per model (VAE, Vanilla GAN, DCGAN, DDPM).",
        "VAE koheren namun soft; Vanilla GAN terstruktur tetapi bertekstur noise; DCGAN "
        "tajam dengan variasi kelas jelas; DDPM tajam dan diverse tanpa mode collapse "
        "yang terlihat. Kontras ini sejalan dengan urutan FID 11,53 < 39,28 < 79,42 < 120,09.",
        max_h=15.5 * cm))
    story.append(PageBreak())
    story.append(fig_unit(
        "Figur4_Latent_Interpolation.png",
        "Figur 4: interpolasi z(a)=(1-a)z_A+a z_B, a in {0,0.2,...,1}, Trouser -> Sneaker.",
        "VAE dan DCGAN menunjukkan transisi semantik mulus (prior kontinu). Vanilla/Sparse "
        "AE melewati daerah berkepadatan rendah (latent voids): sampel antara cenderung "
        "rata-rata kabur / ghosting alih-alih objek valid - bukti visual ketiadaan "
        "regularisasi prior pada AE murni."))
    story.append(FrameBreak())
    story.append(fig_unit(
        "Figur5_Loss_Curves_and_DDPM_Trajectory.png",
        "Figur 5: kurva loss 6 model (G/D terpisah) + trajektori reverse DDPM x_T -> x_0.",
        "AE/VAE/DDPM konvergen monoton; Vanilla GAN osilatif khas adversarial; DCGAN lebih "
        "tenang namun G loss naik di akhir (discriminator dominan). Trajektori bawah "
        "menunjukkan denoising progresif dari noise murni (t=300) ke ankle boot tajam (t=0). "
        "Figur 3 (Bagian III) melengkapi kompilasi ini sebagai bukti batasan klasifier."))
    story.append(NextPageTemplate("one"))
    story.append(PageBreak())

    # ================= Halaman 9-11: Bagian VI =================
    P("Bagian VI - Pembahasan Kritis (Q1-Q6) dan Rekomendasi Industri", H1)
    P("6.1 Q1 - Reconstruction Fidelity vs Generative Quality", H2)
    P("Penalti sparsity lambda*||h||_1 pada Sparse AE menaikkan MSE 0,0069 -> 0,0074 dan "
      "menurunkan SSIM 0,821 -> 0,811 (PSNR 21,60 -> 21,33 dB) sambil menyusutkan ||h||_1 "
      "dari ~130 ke ~17: trade-off kecil dan terkontrol untuk kode yang lebih interpretable. "
      "Namun fidelitas rekonstruksi tidak berkorelasi dengan kualitas generatif: Vanilla AE "
      "rekonstruksi terbaik justru tidak dapat sampling, sedangkan DCGAN (FID 11,53) dan "
      "DDPM (39,28) - tanpa encoder sama sekali - menghasilkan sampel terbaik. VAE berada "
      "di tengah (SSIM 0,727; FID 79,42) karena suku KL menukar detail rekonstruksi dengan "
      "prior yang dapat disampling. Kesimpulan: objective rekonstruksi dan objective "
      "pembentukan distribusi marginal adalah dua sumbu yang berbeda.")
    P("6.2 Q2 - Blurriness vs Sharp Edges & Representasi Fitur", H2)
    P("Minimisasi MSE menghasilkan estimator ekspektasi kondisional:")
    EQ("cond", "ketika p(x|z) multi-modal, rata-rata piksel antar-modus jatuh di antara "
       "objek valid - inilah blur pada AE/VAE. Sebaliknya GAN meminimasi divergensi "
       "Jensen-Shannon D_JS(P_data || P_g): generator dihukum bila sampelnya tidak berada "
       "pada manifold data, sehingga tepi tajam dipertahankan (DCGAN FID 11,53). DDPM "
       "mencapai ketajaman serupa melalui prediksi noise bersyarat yang tidak merata-rata "
       "modus. Pada Sparse AE, penalti L1 mendorong representasi part-based terlokalisasi "
       "(sedikit unit aktif per citra), menjelaskan mengapa rekonstruksinya tetap tajam "
       "meski kode sparse - berbeda dari blur akibat averaging, bukan akibat kapasitas.")
    EQ("js", "dengan M = (P_data + P_g)/2; hukuman ini memaksa sampel berada pada "
       "manifold data, bukan rata-rata antar modus.")
    P("6.3 Q3 - Stabilitas Training, Arsitektur & Generative Trilemma", H2)
    P("Kurva Vanilla GAN osilatif sepanjang 100 epoch (kurva G/D saling kejar), sedangkan "
      "DCGAN lebih tenang berkat inductive bias konvolusi: strided convolution menggantikan "
      "pooling, BatchNorm menstabilkan skala aktivasi, dan LeakyReLU(0.2) mencegah "
      "vanishing gradient pada discriminator - mitigasi mode collapse dan osilasi. "
      "Kenaikan G loss DCGAN di akhir epoch menandakan discriminator mulai dominan, namun "
      "sampel tetap tajam (FID 11,53). DDPM paling stabil karena objective-nya supervised "
      "noise prediction (MSE terhadap noise yang diketahui), bukan permainan adversarial. "
      "Generative trilemma empiris: (a) sample quality - DCGAN 11,53 vs DDPM 39,28; "
      "(b) sampling speed - 4,38 ms vs 2.825 ms per batch 64 (645x lebih lambat); "
      "(c) mode coverage - DDPM menunjukkan keragaman kelas penuh pada Figur 2 tanpa "
      "mode collapse, DCGAN sedikit lebih terbatas variasinya. Tidak ada model yang "
      "memenangkan ketiganya sekaligus.")
    story.append(table([
        ["Model", "Sample quality (FID)", "Sampling speed (ms/b64)", "Mode coverage (Figur 2)"],
        ["VAE", "79,42", "1,20", "penuh, namun soft"],
        ["Vanilla GAN", "120,09", "0,82", "terbatas, bertekstur noise"],
        ["DCGAN", "11,53", "4,38", "baik, variasi sedikit terbatas"],
        ["DDPM", "39,28", "2824,72", "penuh, tanpa mode collapse"],
    ], widths=[2.6 * cm, 4.0 * cm, 4.4 * cm, 5.5 * cm]))
    story.append(Spacer(1, 5))
    P("6.4 Q4 - Struktur Latent Space & Eksperimen Interpolasi", H2)
    P("Interpolasi linear z(a) = (1-a)z_A + a z_B (Figur 4) menguji kekontinuan semantik "
      "ruang laten. VAE (KL terhadap N(0,I)) dan DCGAN (prior kontinu yang dibentuk "
      "adversarial) menghasilkan transisi mulus antar kelas kontras. Vanilla/Sparse AE "
      "tidak memiliki regularisasi aggregate posterior: garis interpolasi melintasi "
      "latent voids berkepadatan rendah sehingga decode menghasilkan rata-rata kabur / "
      "ghosting di alpha menengah. Ini bukti visual langsung bahwa kegunaan interpolasi "
      "datang dari regularisasi prior, bukan dari kapasitas decoder.")
    P("6.5 Q5 - Sintesis Pembuktian Batasan MLP Classifier", H2)
    P("Tiga batasan matematis Bagian III terkonfirmasi berurutan oleh metrik: "
      "(i) information bottleneck P(y|x) menjelaskan hilangnya detail - decoder pada z_MLP beku "
      "(SSIM 0,686) kalah dari AE end-to-end (0,821) meski dim laten sama 64; "
      "(ii) W^T W != I dan ReLU non-invertible - proyeksi balik menghasilkan SSIM 0,053 "
      "dan 0,057 (noise); (iii) ketiadaan prior P(z) - sampling Gaussian menghasilkan blob "
      "(Figur 3 panel 4). Klasifier mencapai 88,44% akurasi justru karena membuang "
      "informasi yang tidak dibutuhkan untuk klasifikasi; informasi itulah yang dibutuhkan "
      "rekonstruksi dan generasi. Maka supervised classifier tidak dapat berfungsi "
      "sebagai generator tanpa objective tambahan pembentuk prior.")
    P("6.6 Q6 - Sintesis Rekayasa & Rekomendasi Deployment Industri", H2)
    P("<b>(1) Deteksi anomali inspeksi manufaktur.</b> Pilih <b>Sparse AE</b>: kode "
      "part-based terlokalisasi membuat reconstruction error terkonsentrasi pada bagian "
      "cacat sehingga sensitivitas terhadap cacat lokal lebih selektif dan interpretable; "
      "biaya fidelitas kecil (SSIM 0,811 vs 0,821). Vanilla AE tetap baseline kuat bila "
      "yang dikejar error global maksimum.")
    P("<b>(2) Filter avatar real-time mobile (&lt;30 ms).</b> Pilih <b>DCGAN</b>: latency "
      "4,38 ms per batch 64 (~0,07 ms/citra) jauh di bawah anggaran 30 ms bahkan untuk "
      "single image, dengan ketajaman tepi terbaik (FID 11,53). VAE (1,20 ms/b64) lebih "
      "cepat namun blur; Vanilla GAN (0,82 ms/b64) tercepat tetapi bertekstur noise "
      "(FID 120,09) sehingga tidak layak untuk produk visual.")
    P("<b>(3) Aset visual fidelitas tinggi industri kreatif.</b> Bila metrik utama FID "
      "dan waktu komputasi ketat: <b>DCGAN</b> (FID 11,53; 4,38 ms/b64). Bila prioritas "
      "mode coverage, keragaman, dan stabilitas training dengan toleransi waktu offline: "
      "<b>DDPM</b> (FID 39,28 namun coverage penuh dan kurva loss paling stabil; "
      "2.825 ms/b64 dapat diamortisasi pada batch rendering offline). Keputusan akhir "
      "bergantung posisi pada generative trilemma (6.3).")
    story.append(table([
        ["Skenario industri", "Pilihan", "Alasan utama"],
        ["Anomali manufaktur", "Sparse AE", "error terlokalisasi pada part cacat; biaya SSIM kecil"],
        ["Avatar mobile <30 ms", "DCGAN", "4,38 ms/b64 + tepi tajam (FID 11,53)"],
        ["Aset kreatif offline", "DDPM", "coverage penuh + stabilitas; latency diamortisasi"],
    ], widths=[4.2 * cm, 2.6 * cm, 9.7 * cm]))
    story.append(Spacer(1, 5))
    P("6.7 Keterbatasan Studi", H2)
    P("Skor FID memakai InceptionV3 pra-latih ImageNet pada citra 28x28 grayscale yang "
      "di-upscale ke 299x299 dan direplikasi ke tiga kanal, sehingga nilai absolutnya "
      "tergeser domain (ranking antarmodel tetap informatif). Setiap model dilatih dengan "
      "satu seed (42) tanpa interval kepercayaan; DDPM dipotong ke T=300 agar feasible "
      "pada satu GPU laptop; kapasitas MLP GAN terbatas (FID 120,09); dan latency diukur "
      "pada laptop GPU yang clock-nya bervariasi terhadap kondisi termal/daya, sehingga "
      "angka milidetik bersifat orde-besaran, bukan jaminan real-time perangkat lain.")
    story.append(PageBreak())
    P("Kesimpulan", H2)
    P("Di bawah satu split dan seed identik, studi ini memisahkan dengan tegas kemampuan "
      "rekonstruksi (AE family), kemampuan sampling (VAE/GAN/DCGAN/DDPM), dan batasan "
      "fundamental model diskriminatif. DCGAN memberi trade-off kualitas-latency terbaik, "
      "DDPM coverage dan stabilitas terbaik dengan biaya sampling tinggi, dan eksperimen "
      "lima tahap membuktikan klasifier MLP tidak dapat menggantikan model generatif.")
    P("Referensi", H2)
    for ref in [
        "[1] Xiao et al. Fashion-MNIST. arXiv:1708.07747, 2017.",
        "[2] Kingma & Welling. Auto-Encoding Variational Bayes. ICLR 2014.",
        "[3] Goodfellow et al. Generative Adversarial Nets. NeurIPS 2014.",
        "[4] Radford et al. DCGAN. ICLR 2016.",
        "[5] Ho et al. Denoising Diffusion Probabilistic Models. NeurIPS 2020.",
        "[6] Heusel et al. FID / Two Time-Scale Update Rule. NeurIPS 2017.",
        "[7] Paszke et al. PyTorch. NeurIPS 2019.",
    ]:
        P(ref, CAP)

    doc = BaseDocTemplate(str(OUT), pagesize=A4,
                          leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=MARGIN, bottomMargin=MARGIN,
                          title="TUGAS_DL_SESI07_2802467382_WISELY_JANSON_HALIM",
                          author="Wisely Janson Halim")
    one = Frame(MARGIN, MARGIN, FRAME_W, FRAME_H, id="one")
    c1 = Frame(MARGIN, MARGIN, COL_W, FRAME_H, id="c1")
    c2 = Frame(MARGIN + COL_W + 0.8 * cm, MARGIN, COL_W, FRAME_H, id="c2")
    doc.addPageTemplates([
        PageTemplate(id="one", frames=[one]),
        PageTemplate(id="two", frames=[c1, c2]),
    ])
    doc.build(story)
    print("Saved", OUT)


if __name__ == "__main__":
    main()
