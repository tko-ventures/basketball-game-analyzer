# Architecture — C4 Visual Views

*Companion diagrams to the canonical [`docs/architecture.md`](architecture.md) (M0–M9 shipped, M10 next, as of 2026-11-24).*

> **Maintenance rule:** [`docs/architecture.md`](architecture.md) is the canonical architecture document. Whenever `docs/architecture.md` changes (active stack, layer DAG/contracts, execution/eval/training mechanics, compute topology, or major subsystems), update this file in the same change so the C4 views stay in sync.

This document presents the current solution from six complementary angles:

1. [C4 Level 1 — System Context](#1-c4-level-1--system-context)
2. [C4 Level 2 — Container Diagram](#2-c4-level-2--container-diagram)
3. [C4 Level 3A — Component Diagram: Pipeline Engine & All 11 Layers (L0–L7)](#3-c4-level-3a--component-diagram-pipeline-engine--all-11-layers-l0l7)
4. [C4 Level 3B — Component Diagram: Training, Provenance & Gold-Leakage Guard](#4-c4-level-3b--component-diagram-training-provenance--gold-leakage-guard)
5. [Deployment & Compute Topology](#5-deployment--compute-topology)
6. [Dynamic / Data-Contract View](#6-dynamic--data-contract-view)

---

## 1. C4 Level 1 — System Context

Actors and external systems surrounding `basketball-game-analyzer`.

```mermaid
flowchart TB
    classDef person fill:#08427B,stroke:#052E56,color:#FFFFFF
    classDef system fill:#1168BD,stroke:#0B4884,color:#FFFFFF
    classDef external fill:#999999,stroke:#6B6B6B,color:#FFFFFF

    analyst["«Person»\nML Engineer / Analyst\nRuns pipeline, trains models,\naudits gold labels, deploys demo"]:::person
    agent["«Person»\nAI Coding Agents\nFirst-pass contact-sheet labelling,\nexperiments, tests & PRs"]:::person
    viewer["«Person»\nCoach / Demo Viewer\nViews 2D court analysis &\ncoach prototype in browser"]:::person

    bga["«Software System»\nBasketball Game Analyzer (BGA)\nOffline PLK broadcast video analysis:\ncourt mapping, detection, team,\ntracking, identity & 2D positioning"]:::system

    yt["«External System»\nYouTube\nPLK highlight clips\n(1080p50 H.264)"]:::external
    plk["«External System»\nplk.pl\nOfficial PLK club rosters\n(2024/25–2026/27)"]:::external
    modal["«External System»\nModal Cloud\nServerless CUDA GPUs (L4/A10G/A100),\nDDP training & shared Volume"]:::external
    colab["«External System»\nGoogle Colab\nEphemeral CUDA GPUs via\ngoogle-colab-cli (Compute Units)"]:::external
    rf["«External System»\nRoboflow Inference & Models\nBaseline NBA weights & inference SDK\n(API key via roboflow secret)"]:::external
    hf["«External System»\nHugging Face / GitHub\nOpen pretrained weights\n(SigLIP, OSNet, SAM 2.1, PARSeq)"]:::external
    vercel["«External System»\nVercel\nStatic hosting for demo site\n(basketball-analyzer.demo.tko.ventures)"]:::external

    analyst -->|"CLI runs, training, labelling UI"| bga
    agent -->|"Runs bga CLI, contact-sheet QA"| bga
    viewer -->|"Watches demo & coach prototype (HTTPS)"| vercel

    bga -->|"Downloads clips (yt-dlp)"| yt
    bga -->|"Scrapes RSC squad JSON"| plk
    bga -->|"Parallel inference & DDP training"| modal
    bga -->|"Single-GPU fine-tunes (Jupyter API)"| colab
    bga -->|"Baseline inference & pretrained checkpoints"| rf
    bga -->|"Pulls open-weight checkpoints"| hf
    bga -->|"Publishes static site (vercel CLI)"| vercel
```

---

## 2. C4 Level 2 — Container Diagram

Runnable applications, cloud workers, and persistent data stores across the local Mac and cloud platforms.

```mermaid
flowchart TB
    classDef person fill:#08427B,stroke:#052E56,color:#FFFFFF
    classDef container fill:#438DD5,stroke:#2E6295,color:#FFFFFF
    classDef cloud fill:#2E6295,stroke:#1B3F63,color:#FFFFFF
    classDef store fill:#1168BD,stroke:#0B4884,color:#FFFFFF
    classDef external fill:#999999,stroke:#6B6B6B,color:#FFFFFF

    user["«Person»\nAnalyst / Agent"]:::person
    viewer["«Person»\nDemo Viewer"]:::person

    subgraph local["«System Boundary: Local Workstation — Apple M1 Max (MPS)»"]
        cli["«Container: CLI & Pipeline»\nbga CLI (src/bga/cli.py)\nPython 3.12, Typer, PyTorch MPS,\nOpenCV, PyArrow, Pydantic"]:::container

        labeler["«Container: Local Web App»\nLabeller Server (tools/labeler/server.py)\nStdlib ThreadingHTTPServer :8765\nPages: /, /court, /boxes, /numbers, /events, /holders, /shotgold"]:::container

        scripts["«Container: Batch Utilities»\nData, QA & Demo Scripts (scripts/)\nClip fetch, roster fetch, contact sheets,\nSAM2 prefill, demo/coach builders"]:::container

        gitdata[("«Data Store: Git-Tracked»\nConfigs, Labels, Rosters, Gold & Docs\nconfigs/, data/labels/, data/rosters/,\ndata/gold/v0..v1, data/fixtures/, docs/")]:::store

        localstore[("«Data Store: Local Gitignored»\nMedia, Weights & Stage Cache\ndata/raw/*.mp4, weights/<id>/,\nruns/<clip>/<layer>/<backend>@<hash8>/")]:::store
    end

    subgraph modal_cloud["«System Boundary: Modal Cloud»"]
        modal_run["«Container: Modal App»\nbga-runner (modal_runner.py)\nParallel per-sequence inference\n(CUDA-only SAM2 or --remote-layer)"]:::cloud

        modal_trn["«Container: Modal App»\nbga-train (modal_train.py)\nSingle-GPU & multi-GPU DDP\nfine-tuning containers"]:::cloud

        modal_vol[("«Data Store: Modal Volume»\nbga-data Volume\n/data/clips, /data/weights,\n/data/train/<weight_id>/")]:::store
    end

    subgraph colab_cloud["«System Boundary: Google Colab»"]
        colab_vm["«Container: Ephemeral GPU VM»\nColab Runtime (colab_train.py)\nChunked upload/download over\nJupyter Contents API; auto-teardown"]:::cloud
    end

    subgraph vercel_edge["«System Boundary: Vercel Edge»"]
        demo_site["«Container: Static Web App»\nDemo & Coach Site\n/ (analysis), /coach/ (synthetic),\n/roadmap/ (milestones)"]:::cloud
    end

    ext_sources["«External Systems»\nYouTube, plk.pl, Roboflow, HF Hub"]:::external

    user -->|"bga run / eval* / train / export / radar"| cli
    user -->|"Labels tracks, court, boxes, numbers (HTTP)"| labeler
    user -->|"Prepares data, builds & deploys demo"| scripts
    viewer -->|"HTTPS"| demo_site

    cli -->|"Reads configs, rosters, gold; logs metrics/costs"| gitdata
    cli <-->|"Reads clips/weights; writes atomic JSON+.npy/Parquet/MP4"| localstore
    labeler <-->|"Reads clips; writes label JSON, .mot.txt, .csv"| gitdata
    labeler -->|"Decodes frames (OpenCV)"| localstore
    scripts <-->|"Reads/writes labels, manifests, rosters"| gitdata
    scripts <-->|"Reads/writes raw clips & demo build"| localstore
    scripts -->|"Downloads clips & rosters"| ext_sources
    scripts -->|"Deploys runs/demo/web (deploy_demo.sh)"| demo_site

    cli -->|"Spawns .map() calls; streams outputs"| modal_run
    cli -->|"Spawns detached/waited training jobs"| modal_trn
    cli -->|"Syncs clips, datasets & weights"| modal_vol
    modal_run <-->|"Reads clips & weights"| modal_vol
    modal_trn <-->|"Reads datasets; commits checkpoints"| modal_vol

    cli -->|"Uploads bga+dataset; runs dispatch; pulls weights"| colab_vm
    modal_run -->|"Pulls pretrained models"| ext_sources
    modal_trn -->|"Pulls pretrained base weights"| ext_sources
    colab_vm -->|"Pulls pretrained base weights"| ext_sources
```

---

## 3. C4 Level 3A — Component Diagram: Pipeline Engine & All 11 Layers (L0–L7)

Internal orchestration of `src/bga/`, the fixed `bga.config.LAYERS` DAG (solid = required upstream, dashed = optional upstream), active leading backends vs. registered alternatives, and downstream consumers.

```mermaid
flowchart TB
    classDef engine fill:#1168BD,stroke:#0B4884,color:#FFFFFF
    classDef layer fill:#85BBF0,stroke:#2E6295,color:#000000
    classDef clipLayer fill:#D4E6F9,stroke:#1168BD,stroke-width:2px,color:#000000
    classDef output fill:#438DD5,stroke:#2E6295,color:#FFFFFF

    subgraph core["«Container: bga Core Engine»"]
        cfg["«Component»\nConfig & Hashing (config.py)\nRunConfig Pydantic schema,\nSHA-256 layer_hash & --set overrides"]:::engine
        reg["«Component»\nBackend Registry (registry.py)\n@register(layer, name)\nLazy plugin loader"]:::engine
        orch["«Component»\nLayer Orchestrator (pipeline.py)\nSequential load->infer->unload;\natomic JSON+.npy cache & meta.json"]:::engine
        reader["«Component»\nFrameReader & Device\n(stage.py, device.py)\nSequential grab <=64f (~4ms/f);\nCUDA->MPS->CPU selection"]:::engine
        cfg --> orch
        reg --> orch
        reader --> orch
    end

    subgraph dag["«Component Group: 11-Layer Pipeline DAG (Solid = Required, Dashed = Optional Upstream)»"]
        L0["«L0: shots — sequence selection»\nActive: gold | clips\nHelper: hist_diff shot-cut detector"]:::layer

        L1["«L1: court — homography»\nActive: rfdetr_keypoint (PLK + RANSAC + smooth4)\nAlt: roboflow_keypoints\nUses fiba_court.json & propagate.py"]:::layer

        L2["«L2: detect — objects & possession»\nActive: rfdetr_torch, plk-r4-l4 (PLK Large @ 1280px MPS;\n6 active classes incl. player-in-possession)\nAlt: roboflow_inference (+ onnx_rfdetr, tiles, nms)"]:::layer

        L3["«L3: team — kit classification»\nActive: siglip_kmeans\nSigLIP + UMAP + KMeans @ 5 fps\nAligned via clip light/dark kit meta"]:::layer

        L3b["«L3b: appearance — re-ID embeddings»\nActive: reid (OSNet x1.0 PLK-tuned,\nfloat16 L2-norm .npy)\nAlt: none"]:::layer

        L4a["«L4a: track — frame-to-frame MOT»\nActive: trackers (BoT-SORT + CMC + PLK re-ID)\nAlt: bytetrack | sam2 (CUDA-only Modal)"]:::layer

        L5a["«L5a: numbers — jersey OCR»\nActive: parseq (PLK-tuned PARSeq probs)\nAlt: crop_cls | smolvlm_ocr"]:::layer

        L4b["«L4b: link — tracklet stitching»\nActive: greedy\nJoins by team, gap, court speed\n+ 5-players-per-team cap"]:::layer

        L5b["«L5b: identity — player resolution (scope: clip)»\nActive: game (cross-shot re-ID clustering +\npooled roster log-prob + Hungarian LSA)\nAlt (shot): covisible_assign | roster_logprob | consecutive"]:::clipLayer

        L6["«L6: positions — 2D FIBA court (x_m, y_m)»\nActive: bbox_bottom (linked-track box foot +\nw>0 guard + clean_paths smooth_window=9)\nComparator: ankles_pose (COCO 17-kp + airborne hold)"]:::layer

        L7["«L7: ball — ball, possession & shots»\nActive: fused, ball-plk-fused-v6-r4 (rim-FP filter +\nDAG DP ball + Viterbi HMM possession +\nrim-trajectory shots, shooter before rim zone,\nfree throws by scene stillness)\nBaseline: skalski (pip + ShotEventTracker)"]:::layer

        L0 --> L1
        L0 --> L2
        L2 --> L3
        L2 --> L3b
        L2 --> L4a
        L1 -.->|"optional"| L4a
        L2 --> L5a
        L4a --> L5a
        L4a --> L4b
        L3 --> L4b
        L3b -.->|"optional"| L4b
        L1 -.->|"optional"| L4b
        L5a -.->|"optional"| L4b
        L4b --> L5b
        L4a --> L5b
        L3 --> L5b
        L5a -.->|"optional"| L5b
        L3b -.->|"optional"| L5b
        L4a --> L6
        L4b -.->|"optional (preferred)"| L6
        L1 -.->|"optional"| L6
        L2 --> L7
        L4a --> L7
        L3 --> L7
        L4b -.->|"optional (preferred)"| L7
        L1 -.->|"optional"| L7
        L6 -.->|"optional"| L7
    end

    subgraph consumers["«Component Group: Export & Evaluation Consumers»"]
        exp_pq["«Component: Export»\nParquet Exporter (export/parquet.py)\nTyped 17-col PLAYER_SCHEMA +\nsibling .ball.parquet (BALL_SCHEMA)"]:::output
        exp_rad["«Component: Export»\nRadar Renderer (export/radar.py)\nStacked broadcast + 2D FIBA court MP4\n(players, ball/holder ring, shot markers)"]:::output
        eval_mod["«Component: Evaluation»\n7 Gold Evaluators (eval/*.py)\neval (TrackEval+churn), eval-court,\neval-detect (COCO), eval-appearance,\neval-numbers, eval-positions,\neval-ball (v1 + shot_gold) -> docs/metrics.md"]:::output
    end

    orch -->|"Executes configured layers"| dag
    L4b --> exp_pq
    L3 --> exp_pq
    L5b --> exp_pq
    L6 --> exp_pq
    L7 -.-> exp_pq
    L4b --> exp_rad
    L3 --> exp_rad
    L5b --> exp_rad
    L6 --> exp_rad
    L7 -.-> exp_rad
    dag -->|"Scored against data/gold/v0 & v1"| eval_mod
```

---

## 4. C4 Level 3B — Component Diagram: Training, Provenance & Gold-Leakage Guard

Dataset preparation, gold-leakage enforcement, Colab/Modal training dispatch, and provenance-backed collection.

```mermaid
flowchart LR
    classDef comp fill:#85BBF0,stroke:#2E6295,color:#000000
    classDef guard fill:#D9534F,stroke:#A94442,color:#FFFFFF
    classDef runner fill:#438DD5,stroke:#2E6295,color:#FFFFFF
    classDef store fill:#1168BD,stroke:#0B4884,color:#FFFFFF

    tcfg[("configs/train/*.yaml\nTrainConfig\nbase weights, manifests,\nrecipe, platform/GPU")]:::store

    subgraph train_local["«Container: bga.train (Local Orchestration)»"]
        train_orch["«Component»\nTraining Orchestrator\n(train/run.py)\nComputes weight_id@hash8,\nresolves --platform/--gpu"]:::comp

        builders["«Component»\nDataset Builders\ncourt_data | detect_data |\nappearance_data | number_data\nVerifies manifest SHA-256"]:::comp

        leak["«Component: Guard»\nGold-Leakage Guard\n(train/leakage.py)\nBlocks any dev or held-out\ngold clip from training"]:::guard

        collect["«Component»\nCollector & Invalidator\n(train/run.py --collect)\nWrites train.json, invalidates\nruns/ cache & Modal volume copy"]:::comp
    end

    subgraph remote_exec["«Remote Training Backends»"]
        colab_r["«Component»\nColab Runner\n(remote/colab_train.py)\nChunked 15MiB transfers,\ndetached trainer + polled logs,\nCU usage, auto-stop"]:::runner

        modal_r["«Component»\nModal Runner\n(remote/modal_train.py)\nDetached / --wait / --resume,\nmulti-GPU DDP, periodic vol commit"]:::runner

        dispatch["«Component: Shared Trainers»\nbga.train.dispatch\ntrain_keypoint | train_detect |\ntrain_appearance (reid_trainer) |\ntrain_numbers (cls_trainer) |\ntrain_parseq (parseq_trainer)"]:::comp
    end

    weights[("weights/<layer>/<backend>/<name>@<hash8>/\ncheckpoint (.pth / .safetensors),\ntrain.json, metrics.csv, result.json")]:::store
    logs[("docs/train.md & docs/costs.md\nValidation scores & billing truth")]:::store

    tcfg --> train_orch
    train_orch --> builders
    builders --> leak
    leak -->|"Clean dataset"| train_orch
    train_orch -->|"--platform colab"| colab_r
    train_orch -->|"--platform modal"| modal_r
    colab_r --> dispatch
    modal_r --> dispatch
    dispatch -->|"Checkpoints + result.json"| collect
    collect --> weights
    collect --> logs
```

---

## 5. Deployment & Compute Topology

Physical execution nodes, memory/cost boundaries, container images, volumes, and secret locations.

```mermaid
flowchart TB
    classDef svc fill:#438DD5,stroke:#1B3F63,color:#FFFFFF
    classDef vol fill:#1168BD,stroke:#0B4884,color:#FFFFFF
    classDef sec fill:#6C757D,stroke:#343A40,color:#FFFFFF

    subgraph mac["«Deployment Node: Apple M1 Max (macOS, Unified Memory, MPS)»"]
        mac_inf["Local Inference (bga run)\n1 heavy model in RAM at a time\nRF-DETR Keypoint, RF-DETR Large 1280px,\nSigLIP, OSNet, PARSeq, Link, Identity, Positions, Ball"]:::svc
        mac_lbl["Local Labeller (:8765)\n+ scripts/serve_demo.py (HTTP Range)"]:::svc
        mac_fs[("Local SSD\ndata/raw/ (39 PLK clips)\nweights/ | runs/ | git repo")]:::vol
        mac_sec["Local Secrets (untracked)\n~/.modal.toml\n~/.config/bga/roboflow.env"]:::sec
    end

    subgraph modal_node["«Deployment Node: Modal Cloud (Linux Containers + NVIDIA GPUs)»"]
        m_img1["bga-runner Container\npython:3.12-slim-bookworm + uv.lock\nBaked SAM 2.1 weights; local HF/RF cache\nL4 GPU (~64ms/f BoT-SORT) or A10G/A100"]:::svc
        m_img2["bga-train Containers\n3 images: rfdetr | reid | parseq\n8 vCPU, 32 GiB RAM, 1–N GPUs (DDP)"]:::svc
        m_vol[("Modal Volume: bga-data\n/data/clips | /data/weights | /data/train")]:::vol
        m_sec["Modal Secret: roboflow"]:::sec
    end

    subgraph colab_node["«Deployment Node: Google Colab (Ephemeral CUDA VM)»"]
        c_vm["Colab GPU Runtime (T4 / L4 / G4 / A100 / H100)\nPreinstalled CUDA PyTorch + delta pip install\nAlways terminated in finally block"]:::svc
    end

    subgraph gha_node["«Deployment Node: GitHub Actions (ubuntu-latest, CPU only)»"]
        gha_ci["CI (.github/workflows/ci.yml)\nuv sync --frozen + pytest (doctests, gold leakage)\n+ pre-commit hooks; no videos, weights or GPU"]:::svc
    end

    subgraph vercel_node["«Deployment Node: Vercel Edge Network»"]
        v_cdn["Static Site CDN\nbasketball-analyzer.demo.tko.ventures\nX-Robots-Tag: noindex, nofollow\nMP4 Cache-Control: 3600s"]:::svc
    end

    mac_inf <-->|"--remote modal / --remote-layer\nSequence results streamed back"| m_img1
    mac_fs <-->|"One-time clip/weight sync"| m_vol
    mac_inf <-->|"bga train --platform modal"| m_img2
    m_img1 <--> m_vol
    m_img2 <--> m_vol
    m_sec -.-> m_img1
    mac_sec -.-> mac_inf
    mac_inf <-->|"bga train --platform colab\n15 MiB chunks via google-colab-cli"| c_vm
    mac_fs -->|"scripts/deploy_demo.sh\n(clean main only)"| v_cdn
    mac_fs -->|"git push / pull request"| gha_ci
```

---

## 6. Dynamic / Data-Contract View

End-to-end transformation of a broadcast sequence from raw 1080p50 MP4 to typed Parquet, radar video, and gold metrics.

```mermaid
flowchart LR
    classDef input fill:#6C757D,stroke:#343A40,color:#FFFFFF
    classDef stage fill:#85BBF0,stroke:#2E6295,color:#000000
    classDef out fill:#1168BD,stroke:#0B4884,color:#FFFFFF

    raw["Raw Video + Metadata\n1080p50 MP4 + label JSON\n(shots, light/dark kit)\n+ PLK Club Rosters"]:::input

    s0["L0: shots\nSequence(clip, shot,\nframes@stride5, meta)"]:::stage
    s1["L1: court\nPer-frame 3x3 H (cm)\nRANSAC + 4-frame median\n+ inlier count"]:::stage
    s2["L2: detect\nPer-frame xyxy boxes,\nconf, class (6 active:\nplayer*, ball, number, ref, rim)"]:::stage
    s3["L3: team + L3b: appearance\nPer-det team (home/away/ref)\n+ float16 L2 embeddings (.npy)"]:::stage
    s4["L4a: track -> L4b: link\nRaw MOT track_ids ->\nstitched 5v5 cap tracklets\n(frames + id map)"]:::stage
    s5["L5a: numbers -> L5b: identity\nCrop class prob vectors ->\nclip-clustered roster log-prob\n+ LSA -> confirmed #jersey"]:::stage
    s6["L6: positions\nBox foot px + w>0 guard\n-> H projection + clean_paths\n-> FIBA (x_m, y_m)"]:::stage
    s7["L7: ball\nDAG DP ball + Viterbi possession\n+ rim-trajectory shots ->\n(x_m, y_m), holder, shots"]:::stage

    pq["Parquet Export\n17-col PLAYER_SCHEMA +\n5-col BALL_SCHEMA\n(.parquet & .ball.parquet)"]:::out
    rad["Radar Video (.mp4)\nAnnotated broadcast +\n2D FIBA court overlay"]:::out
    met["Gold Metrics\ndocs/metrics.md\n(HOTA, IDF1, AP50, paint err,\ntop-1, kf_med_m, poss/shot F1)"]:::out

    raw --> s0
    s0 --> s1
    s0 --> s2
    s2 --> s3
    s1 --> s4
    s2 --> s4
    s3 --> s4
    s2 --> s5
    s3 --> s5
    s4 --> s5
    s1 --> s6
    s4 --> s6
    s4 --> pq
    s5 --> pq
    s6 --> s7
    s6 --> pq
    s7 --> pq
    s4 --> rad
    s5 --> rad
    s6 --> rad
    s7 --> rad
    s1 --> met
    s2 --> met
    s3 --> met
    s4 --> met
    s5 --> met
    s6 --> met
    s7 --> met
```

---

## Layer & Backend Summary

Exact backend contracts live in `src/bga/<layer>/base.py`; run `bga backends` for the live registry. Optional upstreams are in italics (`OPTIONAL_UPSTREAM` and `OPTIONAL_BY_LAYER` in `src/bga/config.py`).

| Layer | Package | Upstream | Active Leading Backend | Registered Alternatives |
|---|---|---|---|---|
| **L0 `shots`** | `src/bga/ingest/` | *(none)* | `gold` / `clips` | `shot_gold` |
| **L1 `court`** | `src/bga/court/` | `shots` | `rfdetr_keypoint` | `roboflow_keypoints` |
| **L2 `detect`** | `src/bga/detect/` | `shots` | `rfdetr_torch` | `roboflow_inference` |
| **L3 `team`** | `src/bga/team/` | `detect` | `siglip_kmeans` | — |
| **L3b `appearance`** | `src/bga/appearance/` | `detect` | `reid` | `none` |
| **L4a `track`** | `src/bga/track/` | `detect`, *(`court`)* | `trackers` | `bytetrack`, `sam2` |
| **L4b `link`** | `src/bga/link/` | `track`, `team`, *(`appearance`, `court`, `numbers`)* | `greedy` | — |
| **L5a `numbers`** | `src/bga/numbers/` | `detect`, `track` | `parseq` | `crop_cls`, `smolvlm_ocr` |
| **L5b `identity`** | `src/bga/identity/` | `link`, `track`, `team`, *(`numbers`, `appearance`)* | `game` *(clip-scoped)* | `covisible_assign`, `roster_logprob`, `consecutive` |
| **L6 `positions`** | `src/bga/positions/` | `track`, *(`link`, `court`)* | `bbox_bottom` | `ankles_pose` *(comparator)* |
| **L7 `ball`** | `src/bga/ball/` | `detect`, `track`, `team`, *(`link`, `court`, `positions`)* | `fused` | `skalski` *(baseline)* |
