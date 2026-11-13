# Notices

## Connectome data

The circuit in this project is built from the **Male CNS v1.0** connectome released by Janelia FlyEM and Google,
licensed **CC-BY** (attribution only, commercial use allowed): https://male-cns.janelia.org/

The files under `data/mb/` (a mushroom body subgraph) and `site/data/` (skeleton meshes and a cell body cloud used
by the dashboard) are derived from that release and are bundled so the container does not have to download the
1 GB source files. They are made by `scripts/extract_mb.py`, `scripts/build_brain_mesh.py` and
`scripts/build_soma_cloud.py` and were changed only by selecting, simplifying and re-encoding the source data.
If you redistribute them, keep this credit.

The model choices (one hemisphere, rate units, dopamine-gated depression) follow published work on the fly
mushroom body, including Aso et al. 2014 and Jiang and Litwin-Kumar 2021. They are our simplifications, not the
authors' models.

## Third-party software

- `dreamdex_core` (Python client for DreamDEX): MIT, Copyright DreamDEX S.A., from
  https://github.com/somnia-chain/dreamdex-bot-kit. Installed as a dependency, not copied.
- `three.js` is loaded by the dashboard from a CDN under the MIT licence.
- The DreamDEX logo geometry in `site/dreamdex-logo.js` comes from the DreamDEX trading app's assets and is used
  with DreamDEX's permission.
