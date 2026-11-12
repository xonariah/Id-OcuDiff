# ID-OcularDiff

Razčlenjena različica priložene kode za učenje generatorja **ID-OcuDiff** ter generiranje ene nove slike očesa in pripadajoče semantične maske.

`train.py` uporablja logiko iz `Arc2EyesTrainTP_glasses(5).py`. `generate.py` uporablja generiranje in končni regularizirani ansambel iz `Arc2Eyes_ensemble_Worldcoin_5x6_70steps_CFG7_1x_ocala.ipynb`, skladno z `semanticne_maske(5).ipynb`. `run_seg2(4).py` je uporabljen kot referenca za identitete, pripravljene podatke in zunanje odvisnosti.

V paket so vključene kode, nastavitve, navodila in prazne ciljne mape. Modeli, slike, ID-vektorji ter izvirni `utils/CLIPTextModelWrapper.py` niso bili priloženi in jih moraš prenesti po spodnjem seznamu.

## 1. Organizacija kode

Vse ukaze zaženi iz korenske mape `ID-OcularDiff`:

```bash
cd ID-OcularDiff
```

| Datoteka | Naloga |
| --- | --- |
| `train.py` | Glavni vhod za učenje; pokliče učno kodo. |
| `generate.py` | Glavni vhod za generiranje slike in maske; obravnava argumente ukazne vrstice. |
| `id_oculardiff/config.py` | Učne nastavitve, poti, prompti in nastavitve vmesnega vzorčenja. |
| `id_oculardiff/training.py` | Izvirna učna zanka, optimizator in izguba. |
| `id_oculardiff/data.py` | `EyeDataset2Tokens`, nalaganje slik, seznam očal in sestavljanje paketov. |
| `id_oculardiff/id_tokens.py` | `IDTokenAdapter`, iskanje mest `id id` in vstavljanje ID-tokenov. |
| `id_oculardiff/sampling.py` | Generiranje vzorčne mreže med učenjem. |
| `id_oculardiff/checkpoints.py` | Shranjevanje izbranih vmesnih modelov. |
| `id_oculardiff/generation.py` | Povezava izbora identitete, generiranja, segmentacije in shranjevanja rezultatov. |
| `id_oculardiff/generation_helpers.py` | Generacijske nastavitve, izbor vzorca, prompta ter nalaganje generatorja in schedulerja. |
| `id_oculardiff/features.py` | Generiranje slike z zajemom U-Net značilnic v istem prehodu. |
| `id_oculardiff/ensemble.py` | Nalaganje desetih MLP-jev, normalizacija značilnic in povprečje verjetnosti. |
| `id_oculardiff/segmentation.py` | Glajenje verjetnosti, povečanje na izhodno ločljivost in izdelava končne maske. |
| `id_oculardiff/morphology.py` | Izvirna morfološka in geometrijska regularizacija. |
| `id_oculardiff/palette.py` | Barvna preslikava razredov in prekrivanje maske čez sliko. |
| `requirements.txt` | Odvisnosti za učenje in generiranje. |
| `.gitignore` | Izključitev podatkov, modelov, rezultatov in lokalnega okolja iz običajnega dodajanja v Git. |

V izvršljivo kodo niso dodani novi komentarji. Obstoječi opisi funkcij, ki so bili že v izvornih datotekah, so ohranjeni.

## 2. Kaj prenesti v posamezne mape

### Datoteke za učenje in generiranje

| Ciljna relativna pot | Kaj preneseš iz obstoječega projekta | Kdaj je potrebno |
| --- | --- | --- |
| `./data/train/` | Vsebino tvoje pripravljene mape `train`: slike `.jpg` ter pripadajoče datoteke `*_1.pt` in `*_2.pt`. | Učenje; tudi privzeti izbor identitete pri generiranju. |
| `./data/testmatrike/` | Vsebino pripravljene mape `testmatrike`, skupaj z obema ID-vektorjema za vsako sliko. | Če želiš generirati iz istih vzorcev kot v priloženem zvezku; uporabi `--data-dir ./data/testmatrike`. |
| `./data/glasses_manual.txt` | Tvoj dejanski seznam imen slik z očali. | Pravilni prompti med učenjem ter samodejni izbor vzorca pri generiranju. |
| `./utils/` | Izvirno mapo `utils`, ki vsebuje `CLIPTextModelWrapper.py`, in vse njene odvisnosti. | Obvezno za učenje in generiranje. |
| `./models/stable-diffusion-2-1/` | Vsebino SD 2.1 snapshot mape `5cae40e6a2745ae2b01ad92ae5043f95f23644d6`. | Obvezno za učenje in generiranje. |
| `./models/ID-OcuDiff/` | Celotno mapo `output_sd2.1_arc2eyes_2id_512_1_4_5e7_e16_1e4_OCALA`, preimenovano v `ID-OcuDiff`. | Generiranje s tvojim že naučenim modelom. Učenje svoj rezultat zapiše na isto privzeto mesto. |
| `./models/segmentation/pixel_mlp_ensemble_10/` | Celotno istoimensko mapo iz `SegmentacijaMLP/annotation_set_8/segmentation_mlp/`. | Obvezno za generiranje pripadajoče semantične maske. |
| `./outputs/generated/` | Ničesar; rezultate tukaj ustvari program. | Izhodne slike, maske, prekrivanja in metapodatki. |

Pri SD 2.1 prenesi dejansko vsebino modela, vključno s podmapami `vae/`, `scheduler/`, `unet/`, `text_encoder/` in `tokenizer/` ter njihovimi konfiguracijami in utežmi. Če je snapshot sestavljen iz povezav na datoteke predpomnilnika, morajo biti v novi mapi dostopne dejanske datoteke, ne samo povezave do stare lokacije.

V `./models/ID-OcuDiff/` morajo ostati `unet/`, `text_encoder/`, `id_adapter.pth` in tokenizer datoteke, ki jih izvirna učna koda shrani neposredno v koren modela. Prenesi celotno izhodno mapo, da ohraniš njihov format. VAE se še naprej nalaga iz osnovnega SD 2.1.

Ansambel potrebuje **vseh enajst datotek**: `ensemble_manifest.pt` ter `models/pixel_mlp_ensemble_model_00.pt` do `models/pixel_mlp_ensemble_model_09.pt`. Datoteke ne združuj ali preimenuj. Manifest vsebuje tudi normalizacijo, izbrane kanale in mesta zajema značilnic. Nalagalnik uporablja datoteke iz nove lokalne podmape `models/` in podpira stara imena poti z Windows ločili.

Izvirni train je bral `glasses.txt`, zvezki pa `glasses_manual.txt`. Novi projekt uporablja `glasses_manual.txt`. Preveri, da ta datoteka vsebuje iste, pravilne oznake za tvoje učne slike; imeni ne dokazujeta, da sta bila stara seznama vsebinsko enaka. V seznamu naj bo po eno ime slike na vrstico, na primer `S5001L01.jpg`, brez poti. Izvirni učni dataset primerja imena dobesedno, vključno z velikostjo črk.

### ResNet, izvirna baza in dodatna obdelava

| Ciljna relativna pot | Kaj preneseš | Vloga v priloženi kodi |
| --- | --- | --- |
| `./models/resnet101_Id-OcuDiff.pth` | `resnet101_e_80_lr_2e-05_best_ariana2.pth`, preimenovan z ohranjeno končnico `.pth`. | Priloženi `run_seg2` in dodatne celice ga uporabljajo za preverjanje podobnosti šarenic. |
| `./data/CASIA-Iris-Thousand/` | Izvirno bazo `CASIA-Iris-Thousand`, če potrebuješ ponovno pripravo ali evalvacijo podatkov. | To je edina izrecno imenovana izvirna baza v priloženih kodah. |
| `./external/iris-recognition-cnn/` | Obstoječi projekt `iris-recognition-cnn`, vključno z `eval_closed_set.py`, definicijami modelov in `scripts/casia_segmentacija.py`. | Izvirna segmentacija/normalizacija in ResNet evalvacija; ta zunanja koda ni bila priložena. |

`train` in `testmatrike` sta imeni tvojih pripravljenih podatkovnih map. Iz prilog ni mogoče zanesljivo navesti dodatnih imen baz. Celotna CASIA baza in ResNet nista potrebna za `train.py` ali `generate.py`, če imaš pripravljene slike in njune ID-vektorje.

Te dodatne mape so rezervirana mesta za tvoj obstoječi postopek priprave in evalvacije. Novi glavni skripti jih ne zaganjata. Zunanja koda ima lahko svoje poti, ki jih je treba urediti ob njenem vključevanju; ni bila predstavljena kot že prilagojena ali preverjena.

Worldcoin ONNX je v zvezkih namenjen primerjavi. Za lastno regularizirano masko ansambla ni potreben. Prav tako za uporabo že naučenega ansambla niso potrebni stari grafi, primerjalne mreže ali celotna mapa ročnih anotacij. Za morebitno ponovno učenje segmentacijskega ansambla obdrži prvotno `annotation_set_8` z metapodatki, latentnimi tenzorji in ročnimi maskami.

## 3. Oblika učnih podatkov in ID-vektorji

V neposredni mapi `./data/train/` mora biti za vsak primer trojica:

```text
S5001L01.jpg
S5001L01_1.pt
S5001L01_2.pt
```

Vsaka `.pt` datoteka mora vsebovati tenzor, ki se splošči v **1024 vrednosti**. Skupaj tvorita dva ID-tokena. Posamezna datoteka `S5001L01.pt` se ne uporablja.

Ohranjena učna koda bere samo `.jpg` datoteke neposredno v `data/train`, brez rekurzivnega pregledovanja podmap. Izpusti imena z `_seg` in slike brez katerega od obeh vektorjev. Generacijski izbor podpira rekurzivne podmape in dodatne slikovne končnice, kot v zvezku.

Učni tok ostaja: pripravljena slika in oba ID-vektorja → prilagoditev ID-tokenov → vstavitev na mesti `id id` → učenje U-Net, besedilnega enkoderja in ID-adapterja. VAE ostaja zamrznjen. Semantične maske niso del izvirne učne izgube generatorja.

**Postopka izdelave `*_1.pt` in `*_2.pt` iz nove surove slike v nobeni od štirih prilog ni.** ResNet evalvacija uporablja `feature_extract_avg_pool`, vendar to ne določa celotnega prvotnega postopka priprave in delitve učnih vektorjev. Zato tu ni izmišljenega nadomestnega ekstraktorja. Za vključitev tudi tega koraka je potrebna tvoja izvirna skripta za ekstrakcijo in razdelitev ID-vektorjev ter njene odvisnosti.

Argument `--image` pri generiranju izbere pripadajoči par že pripravljenih vektorjev. Ne izračuna nove identitete iz same slike in slike ne uporablja kot začetne slike za img2img.

## 4. Okolje

Za prvi preizkus uporabi svoje obstoječe delujoče Python okolje. S tem ohraniš iste različice knjižnic in združljivost izvirnega CLIP ovoja.

V shranjenem izpisu `semanticne_maske(5).ipynb` so navedeni Torch `2.7.0+cu118`, Diffusers `0.36.0` in Transformers `5.4.0`. Celice nameščanja navajajo tudi NumPy `1.26.4` in OpenCV `4.10.0.84`. Te različice so upoštevane v `requirements.txt`, različice drugih odvisnosti pa v prilogah niso določene. Datoteka zato ni popoln preverjen zaklep okolja.

Za nameščanje odvisnosti v izbranem okolju:

```bash
python -m pip install -r requirements.txt
```

CUDA različico Torch in združljivo torchvision ohrani iz svojega delujočega okolja. Sam `requirements.txt` ne določa CUDA distribucije ali vseh dodatnih odvisnosti manjkajoče zunanje kode.

## 5. Učenje

```bash
python train.py
```

Učenje izhaja iz osnovnega modela v `./models/stable-diffusion-2-1/` in rezultat shrani v `./models/ID-OcuDiff/`. Ne gre za nadaljevanje učenja iz že shranjenega `ID-OcuDiff` checkpointa; obnašanje je enako izvirni skripti.

Če je v privzeti izhodni mapi že preimenovani naučeni model in ga želiš obdržati, za novo učenje izberi drugo nadrejeno mapo:

```bash
python train.py --output-dir ./models/retrained/ID-OcuDiff
```

Obstoječa logika zapisovanja na izhodno mesto je ohranjena. Privzeti cilj zato uporabi za novo učenje takrat, ko želiš vanj zapisati nov rezultat.

Ohranjene učne nastavitve:

| Nastavitev | Vrednost |
| --- | --- |
| Daljša stranica slike | 512; dejanska oblika 512 × 384 |
| Batch size | 1 |
| Gradient accumulation | 4 |
| Število epoh | 50 |
| U-Net learning rate | `5e-7` |
| Text encoder learning rate | `1e-6` |
| ID-adapter learning rate | `1e-4` |
| Prediction type | `v_prediction` |
| Mixed precision | `no` |
| `train_ocala` | `True` |
| Izbrani vmesni checkpointi | Epohe 2, 40 in 45 |
| Vmesno vzorčenje | Privzeto izključeno (`sample_every_n_epochs = 0`) |

`train_ocala=True` ohrani tudi učne slike z očali in zanje uporabi izvirni drugačen prompt. Privzeto generiranje brez očal je ločena nastavitev v `generate.py` in tega učnega vedenja ne spreminja.

Vmesni modeli se shranjujejo pod `checkpoints/epoch_NNN/`, dnevniki pod `logs/` in `tb_logs/` znotraj izhodne mape. Če vključiš vmesno vzorčenje, tudi to uporablja **70 korakov in CFG 7**; njegova preostala izvirna logika ostaja enaka.

## 6. Generiranje slike in maske

Privzeto brez očal:

```bash
python generate.py
```

Z očali:

```bash
python generate.py -glasses
```

Deluje tudi zapis `--glasses`. Brez zastavice je `glasses` v negativnem promptu. Z zastavico se pozitivnemu promptu doda `with glasses`, iz negativnega pa se odstrani izraz `glasses`, kot v priloženem generacijskem zvezku.

Za izbor iz prvotne generacijske mape:

```bash
python generate.py --data-dir ./data/testmatrike
python generate.py --data-dir ./data/testmatrike -glasses
```

Za določeno že pripravljeno identiteto:

```bash
python generate.py --image ./data/train/S5001L01.jpg --seed 42
python generate.py --image ./data/train/S5001L01.jpg --seed 42 -glasses
```

Za neposredno podan par vektorjev:

```bash
python generate.py --id1 ./data/train/S5001L01_1.pt --id2 ./data/train/S5001L01_2.pt --seed 42
```

Samodejni izbor uporablja `glasses_manual.txt`: brez zastavice izbira med viri brez očal, z zastavico med viri z očali. Pri eksplicitnem `--image` ali paru `--id1`/`--id2` identiteto določiš sama, zastavica pa še vedno določa želeni pogoj generiranja. Za eksplicitni izbor seznam očal ni obvezen.

Za model iz ločenega novega učenja:

```bash
python generate.py --model-dir ./models/retrained/ID-OcuDiff
```

Preostale možnosti:

```bash
python train.py --help
python generate.py --help
```

Generiranje uporablja ločljivost **640 × 480**, **70 korakov** in **CFG 7**. `--seed` določa ponovljiv naključni izbor in generacijsko seme; dodatni `--denoise-seed` omogoča izrecno nastavitev semena denoisinga. Dejanska uporabljena semena se zapišejo v metapodatke. `--device auto` izbere CUDA, če je dostopna, sicer CPU; dovoljena sta tudi izrecna `cuda` in `cpu`.

Za vsak primer se v `./outputs/generated/` zapišejo:

| Končnica rezultata | Vsebina |
| --- | --- |
| `.png` | Nova generirana slika. |
| `_mask.png` | Barvna regularizirana maska. |
| `_labels.png` | Enokanalna maska z dejanskimi oznakami `0`, `1`, `2`, `3`. |
| `_overlay.png` | Maska, prekrita čez generirano sliko. |
| `_metadata.json` | Semena, prompta, poti, nastavitve in preslikava značilnic. |

Imena temeljijo na identiteti, semenu in pogoju očal. Za ponovitev istega izhoda uporabi drugo izhodno mapo z `--output-dir`; obstoječi rezultati z istim imenom se ne prepisujejo.

Maska nastane iz značilnic, zajetih med istim denoising prehodom kot slika: U-Net značilnice → deset MLP-jev → povprečje softmax verjetnosti → Gaussian glajenje → povečanje verjetnostnih map → morfološka/geometrijska regularizacija. Ohranjena je preslikava učnih korakov ansambla na najbližje Karras sigma vrednosti pri 70 korakih.

Ansambel je ločen že naučen model; `train.py` ga ne uči. Ob zamenjavi generatorja z novo naučeno različico je treba kakovost mask z obstoječim ansamblom ponovno preveriti, saj njegove vhodne značilnice prihajajo iz generatorja.

## 7. Barve mask

Paleta je natančno povzeta po `tab10_label_image`, ki ga uporabljajo zadnje celice `semanticne_maske(5).ipynb`: indeksi `tab10` so `[0, 3, 6, 9]`.

| Oznaka | Razred | Barva | RGB | HEX |
| --- | --- | --- | --- | --- |
| 0 | Ozadje | Modra | `(31, 119, 180)` | `#1F77B4` |
| 1 | Beločnica | Rdeča | `(214, 39, 40)` | `#D62728` |
| 2 | Šarenica | Rožnata iz končne palete tab10 | `(227, 119, 194)` | `#E377C2` |
| 3 | Zenica | Cyan | `(23, 190, 207)` | `#17BECF` |

Stara paleta z vijoličnim ozadjem in tremi rožnatimi odtenki se ne uporablja. Ena rožnata barva za šarenico je prisotna tudi v izrecno navedeni zadnji celici; ohranjena je zato, da je preslikava enaka tvojemu končnemu prikazu. Za nadaljnjo računsko obdelavo uporabi `_labels.png`, kjer so razredi shranjeni brez barvne preslikave.

## 8. Kaj je preverjeno

Izvirna učna koda je preverjena s primerjavo Python AST: vseh 13 izvornih definicij funkcij in razredov je ohranjenih. Razred nastavitev se razlikuje v štirih lokalnih poteh ter dveh zahtevanih vrednostih vzorčenja. Preverjeni so tudi notranji uvozi in globalne reference med učnimi moduli.

Pri generacijski poti so preneseni izvirni scheduler, struktura ID-adapterja, MLP arhitektura, zajem značilnic in regularizacija. Dodana je organizacija v module, vmesnik za en primer in lokalno razreševanje poti do checkpointov. CLI pomoč, sintaksa in prenosljive pomožne funkcije se preverijo brez modelov.

Celotnega učenja, generiranja ali primerjave numeričnih rezultatov tu ni bilo mogoče izvesti: uteži, pripravljene slike/vektorji, izvirni CLIP ovoj in potrebno ML okolje niso bili priloženi. Primerjava kode ne nadomešča preizkusa s tvojimi dejanskimi modeli.

## 9. Priprava za GitHub

V korenu je pripravljen `.gitignore`. Ob običajnem `git add .` v repozitorij ne dodaja podatkov iz `data/`, uteži iz `models/`, rezultatov iz `outputs/` ali kopije zunanjega projekta iz `external/`. Prazne mape so ohranjene z `.gitkeep` datotekami. Izvirna koda `utils/` se lahko doda, ko jo preneseš in preveriš njene odvisnosti.

Lokalni začetek repozitorija:

```bash
git init
git add .
git status
git commit -m "Organize ID-OcularDiff training and generation"
```

Preglej prikazani seznam pred prvim commitom. Oddaljeni GitHub repozitorij v tem koraku ni bil ustvarjen in koda ni bila objavljena.
