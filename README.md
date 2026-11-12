# ID-OcularDiff

## 1. Namestitev odvisnosti

Vse ukaze izvajamo iz korenske mape projekta, kjer sta `train.py` in `generate.py`.

V izbranem Python okolju namestimo odvisnosti:

```bash
python -m pip install -r requirements.txt
```

Za izvajanje na grafični kartici uporabimo PyTorch s podporo CUDA in združljivo različico `torchvision`.

## 2. Priprava podatkov in modelov

Iz obstoječega projekta prenesemo naslednje datoteke:

| Ciljna mapa oziroma datoteka | Kaj prenesemo |
| --- | --- |
| `data/train/` | Učne slike `.jpg` in pripadajoče vektorje `*_1.pt` ter `*_2.pt`. Datoteke postavimo neposredno v to mapo. |
| `data/test/` | Testne slike in pripadajoče vektorje `*_1.pt` ter `*_2.pt`, ki jih uporabimo pri generiranju. |
| `data/glasses_manual.txt` | Seznam imen slik z očali. V vsako vrstico zapišemo eno ime s končnico, brez poti, in ohranimo velikost črk. |
| `utils/` | Izvirni `CLIPTextModelWrapper.py` iz projekta `iris-recognition-cnn` in njegove potrebne lokalne odvisnosti. |
| `models/stable-diffusion-2-1/` | Vsebino osnovnega SD 2.1 snapshota, vključno s podmapami `vae/`, `scheduler/`, `unet/`, `text_encoder/` in `tokenizer/`. |
| `models/ID-OcuDiff/` | Naučeni generator: `unet/`, `text_encoder/`, `id_adapter.pth` in datoteke tokenizerja v korenu mape. Generator lahko ustvarimo tudi z učenjem. |
| `models/segmentation/pixel_mlp_ensemble_10/` | Celotno mapo ansambla iz `SegmentacijaMLP/annotation_set_8/segmentation_mlp/`: `ensemble_manifest.pt` in podmapo `models/` z vsemi desetimi MLP-modeli. |

Za vsako sliko uporabimo že pripravljena ID-vektorja s po **1024 vrednostmi**:

```text
S5001L01.jpg
S5001L01_1.pt
S5001L01_2.pt
```

## 3. Učenje

Učne nastavitve uredimo v `id_oculardiff/config.py` in zaženemo:

```bash
python train.py
```

Učenje začnemo iz osnovnega SD 2.1. Naučeni generator dobimo v `models/ID-OcuDiff/`.

```bash
python train.py --output-dir ./models/retrained/ID-OcuDiff
```

S tem učimo generator; za segmentacijo uporabimo že naučeni MLP-ansambel.

## 4. Generiranje

Za generiranje uporabimo testne vektorje iz `data/test/`, **70 korakov**, **CFG 7** in ločljivost **640 × 480**.

Sliko brez očal ustvarimo z ukazom:

```bash
python generate.py
```

Za sliko z očali dodamo zastavico:

```bash
python generate.py --glasses
```

Določeno identiteto izberemo prek testne slike in njenih pripravljenih vektorjev:

```bash
python generate.py --image ./data/test/S5001L01.jpg --seed 42 --denoise-seed 123
```

Če smo generator shranili v drugo mapo, jo podamo z argumentom `--model-dir`:

```bash
python generate.py --model-dir ./models/retrained/ID-OcuDiff
```

V `outputs/generated/` dobimo generirano sliko, barvno masko, masko z oznakami razredov, prekrivanje in metapodatke. Za ponovitev z istim imenom izhoda določimo drugo mapo z `--output-dir`.

Vse možnosti preverimo z:

```bash
python train.py --help
python generate.py --help
```
