# NASA CARA conjunction test cases (vendored)

53 real satellite conjunctions as CCSDS Conjunction Data Messages (CDMs), plus
NASA's own reference collision-probability values for each.

Copied verbatim from the **NASA CARA Analysis Tools** SDK so that this repo's
validation and experiments are reproducible from a clean clone without needing
the SDK present.

```
Copyright © 2021 United States Government as represented by the
Administrator of the National Aeronautics and Space Administration.
All Rights Reserved.
```

## Licence

Released under the **NASA Open Source Agreement (NOSA) version 1.3** — see
`NOSA_GSC-18848-1.pdf` in this directory, included as the licence requires.

NOSA permits redistribution. Per §3 the obligations are: include the agreement
(done — the PDF above), display the copyright notice prominently (done — above),
and characterise any alterations as Modifications identifying their originator.

**These files are unmodified.** They are byte-for-byte copies of the originals.
No Modification within the meaning of NOSA §3C has been made, so no change-log
is required. If you ever edit them, that changes — record it here.

Source: NASA CARA Analysis Tools, `DataFiles/PcTestCaseCDMs/`
Vendored: 2026-08-08, from SDK commit `0dcb6e2` ("July 2026 code and data update")

Note NOSA §3E: nothing here should be read as a NASA endorsement of this work.

## Contents

| File | What |
|---|---|
| `*.cdm` (53 files) | Real conjunctions, 2020–2023. Filename encodes `<primaryNORAD>_conj_<secondaryNORAD>_<TCA>_<CDM creation time>` |
| `CARA_PcMethod_Test_Conjunctions.xlsx` | NASA's reference Pc values — `Pc2D`, `Nc2D`, `Nc3D`, `PcSDMC` — plus per-case usage-violation flags and orbit parameters |
| `NOSA_GSC-18848-1.pdf` | The licence |

## What is in the CDMs

Each CDM holds, for both objects: ECI state at TCA (km, km/s, EME2000) and a
6×6 position/velocity covariance in the **RTN** frame (m²). Also the combined
hard-body radius (`COMMENT HBR`), miss distance, relative speed, and CARA's
operational `COLLISION_PROBABILITY`.

The covariance is RTN, so it must be rotated to ECI before the two objects'
covariances can be summed — each object has its *own* RTN frame.
`src/tests/cdmParser.jl` does this.

There is **no `OBJECT_TYPE` field**; debris/rocket-body/payload has to be
inferred from `OBJECT_NAME` (`DEB`, `R/B`). `classify_secondary` in the parser
does that.

## Composition

Primaries are all NASA/NOAA operational assets (TERRA, AQUA, ICESAT-2, HST,
NOAA 18/19/21, WORLDVIEW 1–4, CALIPSO, SMAP, …). Secondaries:

| Secondary class | Count | 2D method valid | 2D usage violation |
|---|---|---|---|
| Debris | 35 | 17 | 18 |
| Rocket body | 4 | 1 | 3 |
| Payload (active) | 13 | 5 | 8 |
| Unknown | 1 | 1 | 0 |
| **Total** | **53** | **24** | **29** |

Notable clusters: 7 ICESAT-2 vs. COSMOS 1408 debris CDMs (the Nov-2021 Russian
ASAT event), and 3 repeated TROPICS PATHFINDER vs. LINCS2 conjunctions.

"2D usage violation" is NASA's own flag meaning the assumptions behind the 2D Pc
method do not hold, so its answer is not trustworthy for that case. It tracks
encounter geometry (slow / near-parallel approaches), **not** object type —
which is why debris splits nearly 50/50.

## Use

`src/tests/test_cara_validation.jl` reads this directory by default.

```bash
julia --project=. src/tests/test_cara_validation.jl
```

See `notes/cara_validation_findings.md` for the analysis (note: `notes/` is
gitignored, so that file is local-only).
