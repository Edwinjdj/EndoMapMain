# EndoMapMain

Tools for mapping sulcal imprints on cranial endocasts and comparing them across individuals, developed by Edwin J. de Jager.

## If you use this code, please cite

de Jager, E. J., Risser, L., Mescam, M., Fonta, C., & Beaudet, A. (2022). Sulci 3D mapping from human cranial endocasts: A powerful tool to study hominin brain evolution. *Human Brain Mapping*, 43(14), 4433–4443. https://doi.org/10.1002/hbm.25964

The **Cite this repository** button on the right of this page gives this reference ready to copy (APA or BibTeX).

## What is in this repository

| Folder | What it does |
|---|---|
| `Endocast density map/` | Builds an average endocast from a sample (step 1) and maps every specimen's labelled sulci onto it to make sulcal density (probability) maps (step 2). Two versions: `Endocast density map pipeline` and `Endocast density map pipeline parallel processing` (registers several specimens at the same time). See the `readme` in each folder for installation and use. |
| `Composite Variability/` | Plots from the density-map output (`sulci_report.txt`): occurrence rate and positional variability per sulcus, left and right. |
| `Endocast registration with lineset/` | Aligns endocast meshes to a reference endocast (RANSAC + ICP, Open3D) and applies the same transform to their left and right sulcal curve sets. |

## Contact

Edwin J. de Jager, PALEVOPRIM (UMR 7262, CNRS – Université de Poitiers)
