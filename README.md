# 🎧 Spotify Hybrid Recommendation System

A hybrid music recommender that blends **content-based filtering** (audio features + tags) with **collaborative filtering** (96K users' listening behavior), exposed through a **Streamlit** app, version-controlled with **DVC**, and shipped to production on **AWS** via a fully automated **CI/CD pipeline with Blue/Green deployment**.

Built end-to-end: EDA → modeling → hybridization → dynamic personalization → containerization → cloud infrastructure → zero-downtime deployment.

🔗 GitHub: https://github.com/Chaitanya0604/Spotify-Hybrid-Recommendation-System.git
🔗 DagsHub: https://dagshub.com/Chaitanya0604/Spotify-Hybrid-Recommendation-System

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Part 1 — EDA + Content-Based Filtering](#part-1--eda--content-based-filtering)
- [Part 2 — Collaborative Filtering](#part-2--collaborative-filtering)
- [Part 3 — Hybrid Recommender](#part-3--hybrid-recommender)
- [Part 4 — Dynamic Weighting + Cold Start](#part-4--dynamic-weighting--cold-start)
- [The Streamlit App](#the-streamlit-app)
- [DVC Pipeline](#dvc-pipeline)
- [CI Pipeline (GitHub Actions + AWS S3)](#ci-pipeline-github-actions--aws-s3)
- [CD Pipeline — Dockerize, ECR, EC2](#cd-pipeline--dockerize-ecr-ec2)
- [Blue/Green Deployment (Production Architecture)](#bluegreen-deployment-production-architecture)
- [Troubleshooting Log](#troubleshooting-log)
- [Quick Reference — Commands](#quick-reference--commands)
- [Future Improvements](#future-improvements)
- [Project Status](#project-status)

---

## Overview

This project was built and documented as a public learning series, evolving in four modeling phases and four deployment phases:

| Phase | What it covers |
|---|---|
| Part 1 | EDA, Content-Based Filtering, DVC pipeline |
| Part 2 | Collaborative Filtering on a sparse user-song interaction matrix |
| Part 3 | Hybrid Recommender — combining both engines with normalization |
| Part 4 | Dynamic weighting (diversity slider) + cold-start routing + session caching |
| CI | Automated testing pipeline (GitHub Actions + DVC + AWS S3) |
| CD | Dockerization, ECR, manual EC2 deployment via CodeDeploy |
| Blue/Green | Production-grade zero-downtime deployment with ALB + Auto Scaling |

---

## Architecture

**Decision flow used by the live app:**

```
User enters song
      ↓
In 30K (filtered) dataset?
  ├── YES → Diversity slider → Full Hybrid Scoring
  └── NO  → In 50K (full) dataset?
              ├── YES → Content-Based only (cold start)
              └── NO  → "Song not found" ❌
```

**Deployment architecture (current — Blue/Green):**

```
GitHub push
  → CI: dvc pull → start app → pytest health check
  → CD: docker build → push to ECR
  → Zip appspec.yml + scripts/ → upload to S3
  → CodeDeploy Blue/Green deployment
       → provisions parallel "green" fleet (Auto Scaling Group)
       → ALB health-checks green fleet
       → reroutes traffic
       → terminates old "blue" fleet (after wait window)
  → App served via ALB DNS (not a direct EC2 IP)
```

---

## Dataset

| Dataset | Size | Used by |
|---|---|---|
| **Music Info** | ~50,000 songs, 21 features (audio features, tags, artist metadata) | Content-Based |
| **User Listening History** | 9.7M rows, 96,000 users, play counts | Collaborative |
| **Filtered Songs** | 30,000 songs (overlap with listening history) | Hybrid |

Source: Million Song Dataset (Kaggle).

**Key EDA findings:**
- `genre` was 55.9% missing — dropped entirely.
- `tags` (e.g. *indie*, *chill*, *late-night*) were only 2.2% missing — used as the primary text signal instead of genre.
- 815 duplicate tracks were found and removed (would have distorted similarity scores).
- 20,000 of the 50,000 songs had **zero** listening history — the root cause of the cold-start problem solved in Part 4.

---

## Part 1 — EDA + Content-Based Filtering

**Core idea:** if two songs sound and feel similar, they should sit close together in feature space. Every song becomes a vector; **cosine similarity** (the angle between vectors, not raw distance) measures closeness — important because features like loudness and danceability live on very different numeric scales.

**Feature engineering** (combined in a single `sklearn` `ColumnTransformer`):

| Feature group | Columns | Encoding |
|---|---|---|
| Already 0–1 scaled | `danceability`, `energy`, `valence`, `acousticness` | `MinMaxScaler` |
| Wide numeric ranges | `duration`, `loudness`, `tempo` | `StandardScaler` |
| Categorical | `artist`, `key`, `time_signature` | `OneHotEncoder` |
| Era signal | `year` | `CountEncoder` (normalized — treats year as era popularity) |
| Text | `tags` | `TF-IDF` (top 85 terms) |

**Output:** a sparse matrix of **50,674 songs × 8,431 features** — large but memory-efficient.

**Strength:** works even for songs nobody has ever listened to — pure audio/metadata signal, no user data required.

---

## Part 2 — Collaborative Filtering

**Core idea:** user listening behavior, not song metadata, decides similarity. Two songs can be unrelated by genre, artist, or tempo, yet still get recommended together because the same audience listens to both.

**Building the interaction matrix:**
- Rows → songs, Columns → users, Values → play counts.
- With ~96,000 users interacting with only a tiny fraction of all songs, the matrix is **extremely sparse**.
- Stored as a **SciPy CSR (Compressed Sparse Row)** matrix — only non-zero values, row pointers, and column indices are kept, instead of wasting memory on millions of zeros.
- **Dask** was used for large-scale preprocessing of the raw listening events before matrix construction.

**Recommendation flow:**
1. Extract the target song's listener vector.
2. Compute cosine similarity against all other songs.
3. Rank by similarity score.
4. Return Top-K recommendations.

**Artifacts produced:** `interaction_matrix.npz`, `track_ids.npy`, `collab_filtered_data.csv`

**Limitation:** songs with no user interactions are invisible to this engine — the cold-start problem that motivated the hybrid approach in Part 3.

---

## Part 3 — Hybrid Recommender

Combining the two engines surfaced three real data problems before any model code could run:

### Problem 1 — Shape mismatch
Content-Based ran on 50K songs; Collaborative ran on 30K (only songs with listening history). The two score arrays couldn't be added directly — different sizes.
**Fix:** use the **Filtered Songs dataset (30K)** for both engines.

### Problem 2 — Index mismatch
Both arrays had 30K scores, but index 0 in content-based didn't correspond to index 0 in collaborative — scores were silently misaligned, producing wrong combined results.
**Fix:** sort the Filtered Songs dataset by `track_id` **before** feature transformation, guaranteeing identical row order across both engines.

### Problem 3 — Scale mismatch
Content-based scores clustered near 1 (large); collaborative scores were tiny (sparse-matrix similarity). Content dominated the blend even at a 20% weight.
**Fix:** **Min-Max normalization** on both score arrays, scaling each to 0–1 so the weighting actually behaves as intended.

### The hybrid equation (Part 3, fixed weights)

```
final_score = 0.3 × content_norm + 0.7 × collab_norm
```

Collaborative is weighted higher — human listening patterns are treated as a richer signal than audio features alone.

---

## Part 4 — Dynamic Weighting + Cold Start

Part 3's hybrid worked, but had three real limitations:

### Problem 1 — Hard-coded weights
The 0.3 / 0.7 split was fixed for every user, regardless of what they actually wanted.

**Fix — dynamic weights via a diversity slider** (`W1 + W2 = 1`, always):

| Diversity | content weight | collab weight | Behavior |
|---|---|---|---|
| 1 | 0.9 | 0.1 | Explore by sound |
| 5 | 0.5 | 0.5 | Balanced |
| 10 | 0.0 | 1.0 | Fully personalized |

### Problem 2 — Cold start: 20,000 songs left out
The hybrid system only covers the 30K filtered dataset. The full catalog has 50K songs — the other 20,000 have zero listening history, so forcing hybrid scoring on them means collaborative contributes nothing useful.

**Fix — three-tier song routing:**

```
Song in 30K dataset?  →  Full Hybrid Scoring
Song in 50K only?     →  Content-Based Filtering only
Song in neither?      →  "Song not found" error
```

As a song accumulates listening history over time, it naturally graduates into the hybrid tier — the architecture handles that transition automatically, with no manual reclassification needed.

### Problem 3 — Slow data loading
Every Streamlit interaction reruns the entire script. Without caching, that meant reloading 2 CSVs, 3 sparse matrices, and 1 NumPy array on **every single click**.

**Fix — load once with `st.session_state`:**

| Click | Behavior |
|---|---|
| First click | Data loads and is stored in session memory |
| Every click after | Reads from memory — no reloading |
| Result | Significantly faster recommendations |

---

## The Streamlit App

The app has evolved across all four parts:

- **v1 (Part 1):** Content-Based only.
- **v2 (Part 2):** Added a dropdown to switch between Content-Based and Collaborative, embedded song previews, and similarity scores shown alongside recommendations.
- **v3 (Part 3):** All three modes in one dropdown — Content-Based | Collaborative | Hybrid — with embedded Spotify previews.
- **v4 (Part 4 — current):** One dropdown, three modes, a **diversity slider** for hybrid blending, and automatic cold-start fallback — no extra steps required from the user.

**Inputs:**
- Song name
- Artist name
- Number of recommendations (5 / 10 / 15 / 20)
- Diversity slider (only relevant in Hybrid mode)

---

## DVC Pipeline

All artifacts are versioned and reproducible — no "it worked on my machine."

**Stages (combined across parts):**

```
data_cleaning → transform_data → interaction_data → transform_filtered_data
```

| Stage | Produces |
|---|---|
| `data_cleaning` | `cleaned_data.csv` |
| `transform_data` | `transformed_data.npz` (content-based features) |
| `interaction_data` | `interaction_matrix.npz`, `track_ids.npy`, `collab_filtered_data.csv` |
| `transform_filtered_data` | `transformed_hybrid_data.npz` |

One command reproduces the entire pipeline end to end:

```bash
dvc repro --force
```

`--force` re-runs every stage even if DVC thinks nothing changed.

DVC's remote storage is **AWS S3** (`s3://cmt-spotify-hybrid-rec-remote`, region `ap-south-1`) — large ML artifacts (matrices, encoded data, pickle files) live there instead of in Git.

---

## CI Pipeline (GitHub Actions + AWS S3)

**Goal:** automatically test the app on every push — a safety net that catches problems before they reach production.

**What it does, in order:**
1. Pulls trained ML artifacts from S3 via `dvc pull`.
2. Starts the Streamlit app on the GitHub-hosted runner.
3. Runs a health-check test confirming the app loaded successfully.
4. Reports pass/fail in the GitHub Actions tab.
5. Stops the app and cleans up.

### AWS setup for CI

| Resource | Value |
|---|---|
| S3 bucket (DVC remote) | `cmt-spotify-hybrid-rec-remote` |
| Region | `ap-south-1` (Mumbai) |
| IAM user (CI/DVC) | `cmt-spotify-hybrid-rec-user` with `AmazonS3FullAccess` |
| GitHub secrets | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |

### DVC remote configuration

```bash
dvc remote add myremote s3://cmt-spotify-hybrid-rec-remote
dvc remote default myremote

# Local-only, gitignored:
dvc remote modify --local myremote access_key_id YOUR_ACCESS_KEY_ID
dvc remote modify --local myremote secret_access_key YOUR_SECRET_ACCESS_KEY
dvc remote modify --local myremote region ap-south-1
```

### CI files

| File | Purpose |
|---|---|
| `.github/workflows/ci-cd.yml` | GitHub Actions workflow; runs on every push to `main`. 8 steps: checkout → setup Python → install deps → authenticate AWS → `dvc pull` → start Streamlit → `pytest` → stop Streamlit. |
| `tests/test_app.py` | Health-check test. Waits for the app to load, then asserts an HTTP GET to `localhost:8000` returns 200. |
| `requirements-ci.txt` | Slim CI-only dependency list (the full `requirements.txt` has 150+ packages, including Windows-only ones that break on Ubuntu runners). |

`requirements-ci.txt` pins:

```
dvc==3.58.0
dvc-s3==3.2.0
streamlit==1.41.1
numpy==2.2.1
pandas==2.2.3
scikit-learn==1.6.0
scipy==1.14.1
pytest==8.3.4
requests==2.32.3
boto3==1.36.1
botocore==1.36.1
s3fs==2024.12.0
pathspec==0.12.1
```

> ⚠️ `pathspec` must stay pinned to `0.12.1` — a newer version breaks DVC with a `_DIR_MARK` import error.

### Committing and triggering

```bash
git add .dvc/config .github/workflows/ci-cd.yml tests/test_app.py requirements-ci.txt
git commit -m "Added S3 remote, CI pipeline, and test script"
git push origin main
```

Push triggers the pipeline automatically — watch it under the repo's **Actions** tab.

---

## CD Pipeline — Dockerize, ECR, EC2

### Project resource reference

| Resource | Value |
|---|---|
| AWS Account ID | `114354607243` |
| AWS Region | `ap-south-1` |
| ECR Repository | `spotify_hybrid_recsys` |
| IAM Deploy User | `hybrid-recsys-deploy` |
| CLI Profile | `hybrid-recsys` |

### 1. Dockerizing

Expected project structure (everything the Dockerfile copies must sit alongside it):

```
your-project/
├── Dockerfile
├── requirements.txt
├── app.py
├── collaborative_filtering.py
├── content_based_filtering.py
├── hybrid_recommendations.py
├── data_cleaning.py
├── transform_filtered_data.py
└── data/
    ├── collab_filtered_data.csv
    ├── interaction_matrix.npz
    ├── track_ids.npy
    ├── cleaned_data.csv
    ├── transformed_data.npz
    └── transformed_hybrid_data.npz
```

Build and test locally:

```bash
docker build -t spotify_hybrid_recsys:latest .
docker run -d -p 8000:8000 --name hybrid_recsys_test spotify_hybrid_recsys:latest
docker ps                              # confirm status "Up"
docker logs hybrid_recsys_test         # check for crashes
# Open http://localhost:8000 and test a real search, e.g. "Love Story" by Taylor Swift
docker stop hybrid_recsys_test && docker rm hybrid_recsys_test
```

### 2. IAM user for deployment

- User: `hybrid-recsys-deploy`, programmatic access only (no console access).
- Initial policy: `AmazonEC2ContainerRegistryFullAccess`.
- Later extended (Phase 3 of deployment) with `AmazonS3FullAccess` and `AWSCodeDeployFullAccess`.

```bash
aws configure --profile hybrid-recsys
aws sts get-caller-identity --profile hybrid-recsys
export AWS_PROFILE=hybrid-recsys   # use for the rest of the session
```

### 3. Push image to ECR

```bash
aws ecr create-repository --repository-name spotify_hybrid_recsys --region ap-south-1 --profile hybrid-recsys

aws ecr get-login-password --profile hybrid-recsys --region ap-south-1 \
  | docker login --username AWS --password-stdin 114354607243.dkr.ecr.ap-south-1.amazonaws.com

docker tag spotify_hybrid_recsys:latest 114354607243.dkr.ecr.ap-south-1.amazonaws.com/spotify_hybrid_recsys:latest
docker push 114354607243.dkr.ecr.ap-south-1.amazonaws.com/spotify_hybrid_recsys:latest
```

### 4. Manual EC2 deployment (initial, in-place)

**One-time AWS Console setup:**
- IAM role `EC2-ECR` (EC2 service role) with `AmazonEC2ContainerRegistryFullAccess` + `AmazonS3ReadOnlyAccess` — lets the instance pull from ECR / read S3 with no stored credentials.
- EC2 instance `Hybrid-Recommender-System-Instance` — Ubuntu AMI, `t2.micro`, IAM role `EC2-ECR`, security group `hybrid-recsys-sg` (SSH 22, HTTP 80, All TCP 0–65535, all from `0.0.0.0/0`).
- S3 deployment bucket: `hybridrecsysdeploymentbucket`.
- CodeDeploy service role: `CodeDeployServiceRole`.
- CodeDeploy application: `hybridrecommendersystem` (compute platform: EC2/On-premises).
- EC2 instance tagged `Environment = hybrid-recsys-prod` (how CodeDeploy targets it).
- Deployment group `hybridrecsysdeploymentgroup` — In-place, `CodeDeployDefault.AllAtOnce`, no load balancer.

**One-time setup on the instance (via SSH):**

```bash
sudo apt update && sudo apt install docker.io -y
sudo systemctl start docker && sudo systemctl enable docker
sudo usermod -aG docker ubuntu        # exit and reconnect after this

sudo apt install unzip curl -y
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install && rm -rf aws awscliv2.zip

sudo apt install ruby-full wget -y
cd /home/ubuntu
wget https://aws-codedeploy-ap-south-1.s3.ap-south-1.amazonaws.com/latest/install
chmod +x ./install && sudo ./install auto
sudo systemctl status codedeploy-agent   # expect: active (running)
```

**Triggering a deployment:**

```bash
zip -r deployment.zip appspec.yml deploy/
aws s3 cp deployment.zip s3://hybridrecsysdeploymentbucket/deployment.zip --region ap-south-1 --profile hybrid-recsys

aws deploy create-deployment \
  --application-name hybridrecommendersystem \
  --deployment-group-name hybridrecsysdeploymentgroup \
  --s3-location bucket=hybridrecsysdeploymentbucket,key=deployment.zip,bundleType=zip \
  --region ap-south-1 \
  --profile hybrid-recsys
```

Two hooks run in sequence: `install_dependencies.sh` (installs Docker, AWS CLI), then `start_docker.sh` (pulls the image from ECR and starts the container). Verify at `http://<EC2-Public-IP>:8000` — use `http://`, not `https://`, since no SSL cert was configured at this stage.

### Automated CI/CD workflow (`.github/workflows/ci-cd.yaml`)

The full pipeline used two separate IAM identities for least-privilege access:

| User | Purpose |
|---|---|
| **S3 user** | `dvc pull` only — read access to the DVC remote bucket. |
| **ECR user** | `hybrid-recsys-deploy` — login + push to ECR. |

GitHub secrets used: `AWS_ACCESS_KEY_ID_S3`, `AWS_SECRET_ACCESS_KEY_S3`, `AWS_ACCESS_KEY_ID_ECR`, `AWS_SECRET_ACCESS_KEY_ECR`, `ECR_REPOSITORY_URI` (this one holds just the repo name, `spotify_hybrid_recsys` — not a full URI; the registry host comes separately from the ECR login step).

```yaml
name: CI-CD

on: push

jobs:
  CI:
    runs-on: ubuntu-latest
    steps:
      - name: Code Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install Packages
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Configure AWS Credentials (S3 / DVC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID_S3 }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY_S3 }}
          aws-region: ap-south-1

      - name: DVC Pull
        run: dvc pull

      - name: Run Application
        run: |
          nohup streamlit run app.py --server.port 8000 &
          sleep 30

      - name: Test App
        run: pytest test_app.py

      - name: Stop Streamlit app
        run: pkill -f "streamlit run"

      - name: Configure AWS Credentials (ECR)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID_ECR }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY_ECR }}
          aws-region: ap-south-1

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build, tag, and push docker image to Amazon ECR
        env:
          REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          REPOSITORY: ${{ secrets.ECR_REPOSITORY_URI }}
          IMAGE_TAG: latest
        run: |
          docker build -t $REGISTRY/$REPOSITORY:$IMAGE_TAG .
          docker push $REGISTRY/$REPOSITORY:$IMAGE_TAG
```

This workflow tests and pushes the image on every push — it does **not** deploy by itself. Deployment was, at this stage, still triggered manually (Phase 4 commands above), or later folded into the Blue/Green workflow described next.

---

## Blue/Green Deployment (Production Architecture)

The deployment pipeline was migrated from a single in-place EC2 instance to a fault-tolerant, horizontally scalable Blue/Green architecture behind an Application Load Balancer.

### Before vs. after

| | In-place (previous) | Blue/Green (current) |
|---|---|---|
| Compute | Single EC2 instance, manually configured | Launch Template auto-provisions every instance |
| Scaling | None | Auto Scaling Group, 1–3 instances across `ap-south-1a` / `ap-south-1b` |
| Traffic | Direct EC2 public IP | Application Load Balancer (health-checked) |
| Deploy mechanism | CodeDeploy in-place | CodeDeploy Blue/Green — parallel "green" fleet, health-validated, traffic rerouted, "blue" fleet terminated after a wait window |

### Key resource names

| Resource | Name |
|---|---|
| Launch Template | `HybridFlexisTemplate` |
| Auto Scaling Group | `HybridFlexisAutoScalingGroup` |
| Load Balancer | `HybridFlexisElasticLoadBalancer` |
| Target Group | `HybridFlexisTargetGroup1` |
| CodeDeploy Application | `hybridrecommendersystem` |
| Deployment Group (Blue/Green) | `HybridAccessDeploymentGroupV2` |
| S3 Deployment Bucket | `hybridrecsysdeploymentbucket` |
| CodeDeploy Service Role | `hybrid-rec-codedeploy-service-role-new` |
| EC2 Instance Role | `EC2-ECR` (a.k.a. `EC2Deploy`) |

### Setup highlights

**`start_docker.sh` corrections for the new architecture:**
- Port mapping changed from `-p 8000:8000` to `-p 80:8000` (the ALB listens on 80, forwards to container port 8000).
- AWS account ID corrected to `114354607243` throughout.
- `docker login` runs with `sudo` so credentials are visible to all subsequent Docker commands run with `sudo`.

**EC2 instance role (`EC2-ECR`) extended with:**
- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonEC2RoleforAWSCodeDeploy`
- `AmazonS3ReadOnlyAccess`

**Launch Template (`HybridFlexisTemplate`)** — Ubuntu AMI, instance type `t2.micro` initially (later revised — see below), key pair `hybrid-recsys-key`, security group `hybrid-recsys-sg`, IAM instance profile `EC2-ECR`. User data script installs Docker, AWS CLI, and the CodeDeploy agent automatically on every boot — replacing what was previously done by hand over SSH.

**Auto Scaling Group:**
- Desired / Min / Max: 1 / 1 / 3
- Target tracking on CPU utilization, target 50%
- AZ distribution: balanced best effort
- ELB health checks enabled, 300-second grace period (accounts for the user-data provisioning script's runtime)

**Application Load Balancer:** internet-facing, HTTP listener on port 80, forwarding to `HybridFlexisTargetGroup1`.

**CodeDeploy deployment group (Blue/Green):**

| Setting | Value |
|---|---|
| Deployment type | Blue/Green |
| Environment | Auto Scaling Group → `HybridFlexisAutoScalingGroup` |
| Blue-Green strategy | Automatically copy Auto Scaling Group |
| Traffic routing | Reroute immediately |
| Termination | Terminate original instances after 1 hour |
| Deployment config | `CodeDeployDefault.AllAtOnce` |

**`appspec.yml`:**

```yaml
version: 0.0
os: linux
hooks:
  ApplicationStop:
    - location: scripts/stop_docker.sh
      timeout: 30
      runas: root
  ApplicationStart:
    - location: scripts/start_docker.sh
      timeout: 300
      runas: root
```

**GitHub Actions CD extension** (added on top of the existing CI: test → build → push to ECR):

```yaml
- name: Zip files for deployment
  run: zip -r deployment.zip appspec.yml scripts/

- name: Upload ZIP to S3
  run: aws s3 cp deployment.zip s3://hybridrecsysdeploymentbucket/deployment.zip

- name: Deploy to AWS CodeDeploy
  run: |
    aws deploy create-deployment \
      --application-name hybridrecommendersystem \
      --deployment-config-name CodeDeployDefault.AllAtOnce \
      --deployment-group-name HybridAccessDeploymentGroupV2 \
      --s3-location bucket=hybridrecsysdeploymentbucket,key=deployment.zip,bundleType=zip \
      --file-exists-behavior OVERWRITE \
      --region ap-south-1
```

The existing `hybrid-recsys-deploy` IAM credentials (configured earlier in the same job for ECR) carry over to these steps — no separate `aws configure` calls needed.

---

## Troubleshooting Log

A consolidated record of real issues hit during CI/CD and Blue/Green setup, with root cause and fix for each.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Docker build fails compiling `pywinpty` (Rust error: `cannot find Win32 in windows`) | `pywinpty` is Windows-only; got into `requirements.txt` via a full local `pip freeze` that also included Jupyter, JupyterLab, Dask, distributed, Celery — none needed at runtime | Removed `pywinpty==2.0.14` from `requirements.txt`; CI uses a slim `requirements-ci.txt` |
| 2 | DVC crashes with a `pathspec` `_DIR_MARK` import error | DVC `3.58.0` requires `pathspec==0.12.1` exactly | Pin `pathspec==0.12.1` |
| 3 | `boto3 ClientArgsCreator` error on `dvc push` | Mismatched local `boto3`/`botocore`/`s3transfer` versions | `pip install --upgrade boto3 botocore s3transfer`, then retry |
| 4 | `dvc pull` fails — no credentials | `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` GitHub secrets missing or misnamed | Confirm both secrets exist with exact names in repo Settings → Secrets |
| 5 | CI test fails — connection refused | Streamlit took longer than the scripted wait to load datasets/matrices | Increase `sleep 60` to `sleep 90` (or the equivalent in the test script) |
| 6 | `docker push` fails — "tag does not exist"; then account ID mismatch | Image was never tagged with the full ECR path; separately, an old account ID (`891377050051`) from a prior project's script was still in use instead of `114354607243` | Re-tagged and re-ran all ECR commands under the correct account ID; updated `start_docker.sh` and other deployment scripts accordingly |
| 7 | CodeDeploy console repeatedly rejects the service role for Auto Scaling permissions, even with `AWSCodeDeployRole`, `AmazonEC2FullAccess`, `AutoScalingFullAccess`, `ElasticLoadBalancingFullAccess` attached and a correct trust policy | AWS Console's deployment-group editor validates the role through a stricter internal check than actual IAM policy evaluation (known inconsistency) | Created the deployment group via AWS CLI from CloudShell instead of the console — succeeded immediately, producing `HybridAccessDeploymentGroupV2` |
| 8 | `aws deploy create-deployment` still fails with the same Auto Scaling permission error, even via CLI | Broad managed policies weren't enough; CodeDeploy's Blue/Green validator specifically needed `iam:PassRole` + `ec2:RunInstances` + `ec2:CreateTags` | Attached an inline policy granting those three actions to the service role |
| 9 | Deployment completes all lifecycle hooks, but app is unreachable; target group shows the instance unhealthy | `appspec.yml` referenced `deploy/scripts/start_docker.sh` (old in-place path), but the GitHub Actions workflow zipped only `scripts/` | Corrected `appspec.yml` to reference `scripts/start_docker.sh` and `scripts/stop_docker.sh`, matching the zip step |
| 10 | Script runs, but container still doesn't start — `pull access denied... no basic auth credentials` | `docker login` ran with `sudo` (credentials saved under `/root/.docker/config.json`), but `docker pull`/`docker run` ran without `sudo` as the `ubuntu` user, which couldn't see root's saved credentials | Added `sudo` consistently to every Docker command in `start_docker.sh` |
| 11 | Deployment appears to hang at "Terminate Original Instances" | Expected behavior — deployment group configured with a 1-hour termination wait window for manual rollback | New instance was already healthy and serving traffic; left to auto-complete or terminated manually via the console |
| 12 | App loads, but clicking "Get Recommendations" returns 502 Bad Gateway, then the app stops responding | `t2.micro` (1 GB RAM) ran out of memory loading pandas DataFrames + SciPy sparse matrices and computing similarity on top — container killed by the Linux OOM killer (exit code 137) | Immediate: added a 2 GB swap file. Permanent: revised the Launch Template's instance type from `t2.micro` to `t3.small` (2 GB RAM), then triggered an ASG instance refresh |

---

## Quick Reference — Commands

| Command | What it does |
|---|---|
| `dvc repro --force` | Re-runs all pipeline stages and regenerates all artifacts |
| `dvc push -r myremote` | Uploads artifacts to S3 bucket |
| `dvc pull` | Downloads artifacts from S3 (what CI runs) |
| `dvc status -c` | Confirms S3 is in sync with local |
| `dvc remote list` | Shows configured remotes and which is default |
| `docker build -t spotify_hybrid_recsys:latest .` | Builds the Docker image |
| `docker run -d -p 8000:8000 --name hybrid_recsys_test spotify_hybrid_recsys:latest` | Runs the image locally for testing |
| `docker tag ... / docker push ...` | Tags and pushes the image to ECR |
| `aws sts get-caller-identity --profile hybrid-recsys` | Confirms which AWS identity the CLI is using |
| `aws deploy create-deployment ...` | Triggers a CodeDeploy deployment |
| `sudo systemctl status codedeploy-agent` | Confirms the CodeDeploy agent is running on an EC2 instance |
| `git add . && git commit -m "msg" && git push origin main` | Stages, commits, and pushes — triggers CI/CD |

---

## Future Improvements

- 📊 **Precision@K and Recall@K** — proper ranking-quality evaluation metrics, to move beyond "does this look reasonable" toward measurable recommendation quality.
- 🤖 **Matrix Factorization with SVD** — a more principled collaborative-filtering approach than raw cosine similarity on the interaction matrix.
- 🧠 **Neural, graph-based, and ranking-based models** — explore deep learning and graph approaches as the dataset and infrastructure mature.

---

## Project Status

- ✅ EDA complete; cold-start problem identified early
- ✅ Content-Based Filtering (50K songs, TF-IDF + scaled audio features)
- ✅ Collaborative Filtering (96K users, sparse CSR interaction matrix)
- ✅ Hybrid Recommender (shape/index/scale mismatches resolved, fixed-weight blend)
- ✅ Dynamic weighting via diversity slider + three-tier cold-start routing + session-state caching
- ✅ Streamlit app v4 — single dropdown, three modes, diversity slider, automatic cold-start fallback
- ✅ DVC pipeline — 4 stages, fully reproducible, S3-backed remote
- ✅ CI pipeline — GitHub Actions, automated health-check testing on every push
- ✅ Dockerized application — image builds and runs correctly, verified locally
- ✅ Image pushed to Amazon ECR under the correct AWS account
- ✅ Manual CD pipeline — EC2 + CodeDeploy (in-place) — verified working end to end
- ✅ Automated CI/CD — GitHub Actions tests, builds, and pushes to ECR on every push
- ✅ Blue/Green migration — Launch Template, Auto Scaling Group, Application Load Balancer, CodeDeploy Blue/Green deployment group
- ✅ Production capacity fixed — `t2.micro` → `t3.small` after OOM crashes under real workload
- ⏳ CI/CD pipeline for the recommender's evaluation metrics (Precision@K, Recall@K) — not yet started
- ⏳ SSL/HTTPS — not yet configured; app currently served over HTTP only via the ALB DNS name

---

*This README consolidates the project's CI/CD and deployment documentation (Sessions 7–10) together with the public Spotify Recommender Series (Parts 1–4).*