# StructureCompare

Verktøy for kvantitativ sammenligning av RT-struktursett (DICOM RS-filer) mot en fasit-konturering. Beregner geometriske likhetsmål for evaluering av AI-genererte eller manuelt tegnede stråleterapi-konturer.

## Resultater

Resultatene skrives til en Excel-fil i langt format med én rad per pasient/struktur/plan, med følgende kolonner:

| Kolonne |
|---|
| DICE coefficient (3D) | 
| Hausdorff distance (2D) [mm] | 
| Hausdorff 95 percentile (2D) [mm] | 
| Jaccard index (3D) |
| Average Symmetric Surface Distance (2D) [mm] |
| Absolute/Difference volume (3D mask) | 
| Center of mass (3D difference) [mm] |

Benytter pakken [medpy](https://loli.github.io/medpy/index.html) til å beregne de ulike målene.

## Mappestruktur for inndata

Hver pasientmappe må inneholde:
- Én eller flere `RS*.dcm`-filer som skal evalueres
- Én `RS*.dcm` fasit-fil — identifisert ved at filnavnet inneholder `bkn`

CT-bildefiler (`CT*.dcm`) i samme mappe brukes for pikselstørrelse og koordinatmapping.

## Kjøreskript

| Skript | Beskrivelse |
|---|---|
| `compare3D.py` | Batch-analyse av en nummerert serie testpasienter (`zzART_Head1` … `zzART_Head20`) |
| `compare3D_mtek.py` | Batch-analyse med glob-mønster (f.eks. MTEK / SyngoTest-data) |
| `compare3D_proradnor.py` | Enkeltmappe-analyse (Proradnor-datasettet) |

Endre variabelen `base` / `folder` / `folder_str` øverst i det aktuelle skriptet til å peke på dine data, og kjør deretter:

```bash
python compare3D.py
```

Resultatene lagres i `Output/` (eller arbeidsmappen) som `<dato>_<label>.xlsx`.

## Prosjektstruktur og klassebeskrivelser

```
analysis/
    patient.py      # Patient-klassen — orkestrerer lasting og metrikk-beregning
    structures.py   # Structures-klassen — parser DICOM RS-filer til 3D-masker
    export.py       # Skriver metrikk-dict til Excel-fil
compare3D.py            # Inngangspunkt — hodeserie batch
compare3D_mtek.py       # Inngangspunkt — MTEK / glob batch
compare3D_proradnor.py  # Inngangspunkt — enkeltmappe
```

### `Structures` (analysis/structures.py)

Håndterer alt på DICOM-filnivå:

- Leser alle `RS*.dcm`-filer i pasientmappen og skiller fasit fra sammenligningsfiler basert på `bkn`-markøren i filnavnet
- Bygger opp oppslagstabeller for ROI-navn → ROI-nummer og ROI-navn → indeks i `ROIContourSequence`
- Normaliserer ROI-navn til lowercase og håndterer kjente navnevariasjoner (f.eks. `constrictmusc_pharynx` → `pharynxconstrict`)
- Leser CT-metadata (SOPInstanceUID, z-posisjon, piksel-origo) for å kunne kartlegge konturdata til riktig snitt-indeks
- `loadStructureMask3D()` konverterer polygon-konturdata fra DICOM til en binær 3D numpy-maske (`columns × rows × slices`) ved å tegne polygoner vha. `skimage.draw.polygon`

### `Patient` (analysis/patient.py)

Bruker `Structures` og beregner metrikker på tvers av alle strukturer og planer:

- Oppretter én `Structures`-instans for fasit og én for sammenligningsfilene
- Finner felles strukturer på tvers av alle RS-filer og håndterer navnemapping (f.eks. `femur_head_l` → `femuralhead_l`) slik at strukturer med ulike navn kan sammenlignes
- Spesialbehandling for `spinalcord`: finner lengste felles z-rekkevidde på tvers av alle RS-filer, slik at en avkutting på ulike z-nivåer ikke kunstig øker feilen
- `build_metrics_for_structure()` henter 3D-masker for fasit og alle sammenligningsfiler, og beregner: DICE, Jaccard, Hausdorff, HD95, ASSD, volumforskjell og euklidsk avstand mellom tyngdepunkter

## Installasjon

```bash
pip install -r requirements.txt
```

Avhengigheter: `pydicom`, `numpy`, `scikit-image`, `scipy`, `medpy`, `seg-metrics`, `pandas`, `matplotlib`, `tqdm`
