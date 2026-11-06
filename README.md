# RoboSAPIENS Temporal Policy Selection: An INTO-CPS Webots Case Study

**Anomaly-aware self-adaptive navigation for multimodal mobile robots.**

This repository provides a Webots case study for the INTO-CPS ecosystem, demonstrating how a cyber-physical robotic system can continue operating when some sensing inputs become unreliable. It combines independent learned anomaly classifiers, four observation-specific PPO navigation policies, and a temporal **Monitor–Analyze–Plan–Execute over Knowledge (MAPE-K)** controller that selects a policy compatible with the current sensor-health state.

## Relation to INTO-CPS

The case study illustrates the interaction of sensing, health monitoring, decision-making, and robot control in a closed loop:

1. A simulated mobile robot provides RGB camera, LiDAR, and task-state observations.
2. Learned, modality-specific anomaly classifiers estimate whether the image and LiDAR streams are reliable.
3. Temporal health reasoning confirms fault onset and recovery and propagates the LiDAR-to-task-state dependency.
4. The MAPE-K controller selects a compatible trained navigation policy, while a minimum dwell interval restricts unnecessary switchback.
5. If neither image nor LiDAR supports a navigation policy, the robot executes the no-motion fallback action.

The repository supports the full workflow: capture nominal sensor sequences; synthesize diverse training anomalies; train and calibrate the image and LiDAR classifiers; train each PPO navigation policy from scratch; run standalone policy inference; and execute the integrated MAPE-K controller. All controller, configuration, world, PROTO, and texture assets remain in the same Webots project layout as the [research implementation](https://github.com/Pavlos-Tosidis/temporal-policy-selection).

**Associated manuscript:** *Temporal Deep Reinforcement Learning Policy Selection for Mobile Robot Navigation under Sensor Anomalies* — Pavlos Tosidis, Nikolaos Passalis, and Anastasios Tefas. **Submitted to Applied Soft Computing.** The manuscript does not yet have a published citation or DOI.

## Case study scenario

The Webots scenario models a TurtleBot navigating toward a person while avoiding obstacles. The robot uses camera, LiDAR, and localization-derived task-state observations. Controlled camera occlusions/failures and LiDAR corruptions/failures can be introduced to examine how the control system responds to degraded sensing. The policy library contains four independent PPO agents trained for different observation subsets; the selector does not retrain policies at runtime.

## Features

- **End-to-end anomaly-classifier lifecycle:** capture nominal camera and LiDAR sequences in Webots, synthesize sensor faults, construct modality-specific datasets with cross-modal negatives, train two classifiers, calibrate detection thresholds, and load them for runtime inference.
- **Four PPO navigation policies:** train one policy for each observation subset with a common training entry point and shared PPO hyperparameters.
- **Independent policy inference:** deploy any trained policy, using the appropriate single-array or dictionary observation format, under nominal or injected sensor-anomaly conditions.
- **Temporal MAPE-K execution:** monitor learned anomaly scores, confirm fault onset and recovery, propagate input dependencies, select among admissible policies, constrain preference-driven switching, and stop the robot when no policy remains admissible.
- **Configurable research workflows:** command-line parameters for dataset collection, training, inference, anomaly conditions, seeds, device selection, and runtime logging.

### Policy library

| Policy key (`--modality`) | Required observations | Intended role |
| --- | --- | --- |
| `image_lidar_world_info` | Camera, LiDAR, task-state information | Preferred policy under nominal sensing |
| `image_lidar` | Camera, LiDAR | Alternative policy without task-state information |
| `image` | Camera | Navigation when LiDAR is unavailable |
| `lidar_world_info` | LiDAR, task-state information | Navigation when the camera is unavailable |

The policy-selection algorithm computes which policies are **admissible** from their declared observation requirements; it does not use a hard-coded fault-to-policy lookup.

## Project layout

The controller package follows the usual Webots project layout and works alongside the project's existing worlds, PROTOs, and other assets.

```text
<webots-project>/
├── worlds/
├── protos/
├── controllers/
│   └── policy_selection/
│       ├── policy_selection.py         # Command-line / Webots entry point
│       └── selection/
│           ├── webots_env.py          # Robot, observation, action and reward interface
│           ├── modalities.py          # Observation subsets and routing
│           ├── policy_models.py       # PPO feature extractors
│           ├── train_policy.py        # PPO training
│           ├── dataset.py             # Webots capture and classifier datasets
│           ├── corruptions.py         # Synthetic RGB/LiDAR anomalies
│           ├── detectors.py           # Image Patch-MIL and LiDAR Conv1D
│           ├── train_detector.py      # Classifier training and calibration
│           ├── monitor.py             # Learned anomaly scores / thresholds
│           ├── mapek.py               # Temporal health and policy selection
│           ├── infer.py               # Policy and MAPE-K execution
│           └── runtime.py             # Environment and checkpoint loading
├── configs/
│   └── policy_library.example.json
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

Datasets, checkpoints, TensorBoard logs, and inference traces are generated locally in the directories specified by the commands below.

## Requirements and installation

- **Webots** with Python-controller support and a compatible simulation world containing the robot, camera, LiDAR, and required world objects.
- **Python 3.10+**, with a Python version supported by the installed Webots release and ML dependencies.
- **PyTorch**, **Stable-Baselines3**, **Gymnasium**, **Ultralytics**, and the other dependencies in `requirements.txt`.
- An appropriate **CPU or CUDA** PyTorch build. GPU execution is recommended for the image-based networks.

From the Webots project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# If CUDA is required, install the matching PyTorch build for your system first.
pip install -r requirements.txt
```

Install PyTorch for your platform using the [official installation instructions](https://pytorch.org/get-started/locally/). Webots supplies the `controller` Python module; it is **not** a PyPI dependency. In Webots, configure the **Python command** to use your virtual environment's interpreter if necessary.

The navigation task uses an Ultralytics YOLO person detector (`yolo11n.pt`) for its **detection action**. This model is separate from the anomaly classifiers described below. Make its weights available to the controller, or allow Ultralytics to obtain them on first use.

### Webots controller setup

Configure the robot as a **Supervisor** in your `.wbt` world and ensure the device names and DEF nodes expected by `selection/webots_env.py` match your scene. The controller can be run in either of two ways:

**Option A — External controller (recommended for command-line workflows).** Set the robot's controller to `<extern>`, open the world in Webots, and start the simulation. From the project root:

```bash
export WEBOTS_HOME=/path/to/your/webots/installation
RUN="$WEBOTS_HOME/webots-controller controllers/policy_selection/policy_selection.py"
$RUN --help
```

Use `$RUN` for any command that interacts with Webots. Each collection, PPO training, or inference run requires a connected controller session; reconnect the external controller or restart the world before starting another run.

**Option B — Native Webots controller.** Set the robot's controller to `policy_selection` and provide the desired command and arguments through `controllerArgs` (one argument per list item):

```webots
controller "policy_selection"
controllerArgs ["collect" "--output" "datasets/nominal_v1" "--episodes" "300"]
```

All example paths below are relative to the **project root**. Run from that directory when using the external controller. Absolute paths can also be used.

## 1. Collect nominal sensor observations

A pretrained PPO policy is **not required**. The default collector resets the simulation for each episode and performs a deterministic rotation/repositioning routine to gather observations from multiple viewpoints. Each episode is stored as a separate sequence.

```bash
$RUN collect \
  --output datasets/nominal_v1 \
  --episodes 300 \
  --steps-per-episode 150 \
  --stride 5 \
  --seed 42
```

The collector saves `datasets/nominal_v1/manifest.json` and one compressed `.npz` file per episode under `datasets/nominal_v1/sequences/`. Each sequence contains camera images, LiDAR stacks, task-state observations, and capture timesteps. The manifest records provenance and collection settings.

**Optional: collect observations using a trained PPO policy** instead of the default scripted movement:

```bash
$RUN collect \
  --output datasets/nominal_policy_v1 \
  --checkpoint outputs/policies/full/model.zip \
  --policy-modality image_lidar_world_info \
  --episodes 300 \
  --steps-per-episode 150 \
  --stride 5 \
  --seed 43
```

These commands gather **nominal** sensor sequences. Anomalies for classifier learning are synthesized from the clean captures during the next stage. The captured camera arrays use the original Webots camera channel order; do not change channel order or sensor normalization between data collection, training, and deployment.

## 2. Train the anomaly classifiers

Classifier training runs **offline** and does not require a Webots connection. Both models share the same nominal collection and create their own labeled inputs.

### Image anomaly classifier — Patch-MIL

```bash
python controllers/policy_selection/policy_selection.py train-detector \
  --modality image \
  --data datasets/nominal_v1 \
  --output outputs/detectors/image \
  --epochs 20 \
  --batch-size 4 \
  --val-fraction 0.2 \
  --seed 42 \
  --device auto
```

The image classifier uses the **503,137-parameter Patch-MIL** architecture. Synthetic RGB anomalies include rectangles, ellipses, polygons, blobs, and strips at varied locations, with black, gray, or random-color contents and target affected-area fractions of **5%, 10%, 20%, and 35%**. Full-image failure is also included.

### LiDAR anomaly classifier — Persistence Conv1D

```bash
python controllers/policy_selection/policy_selection.py train-detector \
  --modality lidar \
  --data datasets/nominal_v1 \
  --output outputs/detectors/lidar \
  --epochs 20 \
  --batch-size 32 \
  --val-fraction 0.2 \
  --seed 42 \
  --device auto
```

The LiDAR classifier uses the **591,777-parameter temporal Conv1D** architecture, operating on an eight-frame LiDAR stack. Partial faults corrupt a consistent angular sector across the stack, using sector widths of **5%, 10%, 20%, or 35%** and simulated close-range returns at **0.05, 0.10, 0.20, 0.30, or 0.50 m**. Complete LiDAR failure is also included.

### How the classifier datasets are constructed

The clean sequences are assigned to training and validation groups **before** synthetic corruption, so frames from a single captured episode do not appear in both sets. For each detector, four variants are generated from each nominal frame:

1. **Nominal negative:** the target modality is healthy.
2. **Cross-modal negative:** the other sensor is corrupted, while the target modality remains healthy.
3. **Partial positive:** the target modality contains a localized fault.
4. **Full positive:** the target modality has failed completely.

Augmentations are generated on demand, and the training variants change across epochs. Each detector is optimized with binary cross-entropy; a decision threshold is selected from validation logits by maximizing balanced accuracy.

Saved artifacts:

```text
outputs/detectors/
├── image/
│   ├── image_detector.pt
│   └── metadata.json
└── lidar/
    ├── lidar_detector.pt
    └── metadata.json
```

The checkpoints include the trained parameters and calibrated decision thresholds. The metadata includes dataset sequence identifiers, training settings, classifier architecture, validation metrics, and library versions. Thresholds are recalibrated for each new training run.

### Inspect learned detector outputs in Webots

Reconnect the external Webots controller and run:

```bash
$RUN infer-detector \
  --image-detector outputs/detectors/image/image_detector.pt \
  --lidar-detector outputs/detectors/lidar/lidar_detector.pt \
  --condition partial_lidar \
  --steps 500
```

This prints the image/LiDAR anomaly scores and decisions obtained from the learned detectors. The `--condition` argument injects a controlled sensor condition for running the controller; the detectors do **not** receive the injected condition as an input.

## 3. Train the four PPO navigation policies

Each policy is trained **independently**, using only the observation subset associated with its key. All four runs use the same PPO setup; the observation encoders select the relevant image, LiDAR, and task-state branches.

| PPO setting | Default |
| --- | ---: |
| Environment steps per policy | **15,000,000** |
| Rollout length (`n_steps`) | 2,048 |
| Minibatch size | 128 |
| Discount factor (`gamma`) | 0.99 |
| GAE coefficient (`gae_lambda`) | 0.95 |
| PPO epochs per update | 10 |
| Initial learning rate | `3e-4`, linearly decayed |
| Entropy coefficient | 0.01 |
| Value-function coefficient | 0.5 |
| Target KL | 0.03 |
| Actor/critic MLP | `[256, 128]` per head |

Run the following commands **one at a time**, reconnecting the external Webots controller between runs:

```bash
# Image + LiDAR + task state
$RUN train-policy \
  --modality image_lidar_world_info \
  --output outputs/policies/full \
  --steps 15000000 \
  --seed 3 \
  --device auto

# Image + LiDAR
$RUN train-policy \
  --modality image_lidar \
  --output outputs/policies/image_lidar \
  --steps 15000000 \
  --seed 3 \
  --device auto

# Image only
$RUN train-policy \
  --modality image \
  --output outputs/policies/image \
  --steps 15000000 \
  --seed 3 \
  --device auto

# LiDAR + task state
$RUN train-policy \
  --modality lidar_world_info \
  --output outputs/policies/lidar_world_info \
  --steps 15000000 \
  --seed 3 \
  --device auto
```

Each run creates its own output directory, with `model.zip`, `vecnormalize.pkl`, `metadata.json`, training logs, and periodic checkpoints in `checkpoints/`. Reward normalization is enabled during PPO training; the environment handles input preprocessing and normalization. Keep each policy's original checkpoint and metadata together.

### Navigation task and observations

The Webots adapter implements the original TurtleBot navigation task: approach a target person while avoiding collisions, and invoke the detection action when positioned appropriately.

| Component | Configuration |
| --- | --- |
| Camera observation | `270 × 480 × 3` (`uint8`) |
| LiDAR observation | 8 stacked scans, each padded to 1,200 samples |
| Task-state information | 3 values |
| Episode horizon | 500 control steps |
| Success distance | `0.3 m ≤ d ≤ 2.0 m` |
| Success angular error | `< 20°` |

The six discrete actions have the following meanings:

| Action ID | Operation |
| ---: | --- |
| `0` | Move forward **0.01 m** |
| `1` | Move backward **0.01 m** |
| `2` | Rotate left **1°** |
| `3` | Rotate right **1°** |
| `4` | Run the task-specific person-detection action |
| `5` | No motion (safe fallback) |

The environment adapter contains the task-specific reward, collision, episode termination, and sensor preprocessing logic. The PPO policies remain fixed during inference; the MAPE-K controller changes which policy is executed rather than retraining one online.

## 4. Run an individual policy

Individual policies can be loaded independently, either in nominal conditions or with a simulated anomaly affecting a sensor stream. Inference uses **stochastic PPO action sampling** by default. Add `--deterministic` to use deterministic actions.

Run each example in its own Webots controller session:

```bash
# Full multimodal policy
$RUN infer-policy \
  --modality image_lidar_world_info \
  --checkpoint outputs/policies/full/model.zip \
  --condition normal \
  --episodes 10 \
  --max-steps 500 \
  --output outputs/traces/full_normal

# Image + LiDAR policy
$RUN infer-policy \
  --modality image_lidar \
  --checkpoint outputs/policies/image_lidar/model.zip \
  --condition normal \
  --episodes 10 \
  --max-steps 500 \
  --output outputs/traces/image_lidar_normal

# Image-only policy
$RUN infer-policy \
  --modality image \
  --checkpoint outputs/policies/image/model.zip \
  --condition full_lidar \
  --episodes 10 \
  --max-steps 500 \
  --output outputs/traces/image_full_lidar

# LiDAR + task-state policy
$RUN infer-policy \
  --modality lidar_world_info \
  --checkpoint outputs/policies/lidar_world_info/model.zip \
  --condition full_image \
  --episodes 10 \
  --max-steps 500 \
  --output outputs/traces/lidar_world_full_image
```

The supported values for `--condition` are:

| Condition | Meaning |
| --- | --- |
| `normal` | No injected fault |
| `partial_image` / `full_image` | Camera occlusion / complete image loss |
| `partial_lidar` / `full_lidar` | LiDAR sector corruption / complete LiDAR loss |
| `partial_both` / `full_both` | Simultaneous camera and LiDAR anomalies |

Partial faults are sampled once per episode and held in sensor coordinates until reset. When `--output` is specified, inference stores a JSONL trace inside the named output directory.

## 5. Run temporal MAPE-K policy selection

The full runtime loads the **four pretrained PPO policies** and **two trained anomaly detectors**. At each control decision, it:

1. **Monitors** image and LiDAR observations using the learned detector models and their calibrated thresholds.
2. **Analyzes** anomaly evidence using separate onset/recovery confirmation counters and marks task-state information unreliable if LiDAR is unreliable.
3. **Plans** by constructing the set of policies whose required observations are reliable and choosing the highest-priority admissible policy.
4. **Executes** the selected policy or issues action `5` (no motion) if no policy is admissible.
5. **Maintains knowledge** of health states, policy preferences, active policy, and switch history across control decisions.

The dwell interval constrains transitions **between admissible policies**, especially switchback following recovery. If the active policy becomes inadmissible, a required transition is not delayed by dwell. The selector does not use ground-truth anomaly labels.

### Configure the policy library

```bash
cp configs/policy_library.example.json configs/policy_library.json
```

The supplied configuration has this structure:

```json
{
  "image_lidar_world_info": {
    "model_path": "../outputs/policies/full/model.zip",
    "obs_keys": ["image", "lidar", "world_info"]
  },
  "image_lidar": {
    "model_path": "../outputs/policies/image_lidar/model.zip",
    "obs_keys": ["image", "lidar"]
  },
  "image": {
    "model_path": "../outputs/policies/image/model.zip",
    "obs_keys": ["image"]
  },
  "lidar_world_info": {
    "model_path": "../outputs/policies/lidar_world_info/model.zip",
    "obs_keys": ["lidar", "world_info"]
  }
}
```

Relative `model_path` values are resolved **relative to the JSON file**, not to the shell's working directory. Update the paths if your trained checkpoints are stored elsewhere. The `obs_keys` entries describe the observation subsets; the selector's corresponding requirements and default preference order are implemented in `selection/modalities.py`.

### Normal operation

```bash
$RUN infer-mapek \
  --library configs/policy_library.json \
  --image-detector outputs/detectors/image/image_detector.pt \
  --lidar-detector outputs/detectors/lidar/lidar_detector.pt \
  --condition normal \
  --episodes 10 \
  --max-steps 500 \
  --onset 5 \
  --recovery 10 \
  --dwell 10 \
  --output outputs/traces/mapek_normal
```

### Image anomaly: transition toward LiDAR + task state

```bash
$RUN infer-mapek \
  --library configs/policy_library.json \
  --image-detector outputs/detectors/image/image_detector.pt \
  --lidar-detector outputs/detectors/lidar/lidar_detector.pt \
  --condition partial_image \
  --episodes 10 \
  --max-steps 500 \
  --onset 5 --recovery 10 --dwell 10 \
  --output outputs/traces/mapek_image
```

When image degradation is confirmed, policies requiring the image become inadmissible; `lidar_world_info` remains available while LiDAR remains reliable.

### LiDAR anomaly: transition toward image-only navigation

```bash
$RUN infer-mapek \
  --library configs/policy_library.json \
  --image-detector outputs/detectors/image/image_detector.pt \
  --lidar-detector outputs/detectors/lidar/lidar_detector.pt \
  --condition partial_lidar \
  --episodes 10 \
  --max-steps 500 \
  --onset 5 --recovery 10 --dwell 10 \
  --output outputs/traces/mapek_lidar
```

When LiDAR degradation is confirmed, `world_info` is also treated as unavailable; `image` remains the admissible reduced-input navigation policy.

### Simultaneous anomalies: no-motion fallback

```bash
$RUN infer-mapek \
  --library configs/policy_library.json \
  --image-detector outputs/detectors/image/image_detector.pt \
  --lidar-detector outputs/detectors/lidar/lidar_detector.pt \
  --condition full_both \
  --episodes 10 \
  --max-steps 500 \
  --onset 5 --recovery 10 --dwell 10 \
  --output outputs/traces/mapek_both
```

If **both** monitored modalities are confirmed unreliable, no policy in the configured library is admissible and the controller executes the no-motion **safe fallback (action `5`)**.

The runtime parameters `--onset`, `--recovery`, and `--dwell` express **control decisions**. The defaults are 5, 10, and 10 respectively; they can be changed independently. Switching events are printed to the console; specifying `--output` saves per-step policy, detector, and health information in `trace.jsonl`.

## Configuration and outputs

| Option | Description |
| --- | --- |
| `--device auto` | Use CUDA if available; otherwise CPU. `cpu` and `cuda` may be selected explicitly. |
| `--seed N` | Seed collection, training, or inference where supported. |
| `--steps 15000000` | Total PPO environment steps per policy. |
| `--save-freq N` | PPO checkpoint interval in environment steps (`0` disables periodic saves). |
| `--val-fraction 0.2` | Fraction of captured sequences reserved for classifier validation. |
| `--condition NAME` | Inject an anomaly condition during inference. |
| `--deterministic` | Replace stochastic PPO inference with deterministic action selection. |
| `--output PATH` | Output directory for collections, training runs, or inference traces. |

Output directories for collection and training must **not already exist**; the software prevents silent overwrites. Use a new run name for each experiment. Trained weights and generated datasets are excluded from Git by default. Keep checkpoint files with their metadata and store large model releases separately if distributing pretrained agents.

For the full set of arguments available to any workflow, use:

```bash
python controllers/policy_selection/policy_selection.py --help
python controllers/policy_selection/policy_selection.py train-detector --help
python controllers/policy_selection/policy_selection.py train-policy --help
python controllers/policy_selection/policy_selection.py infer-mapek --help
```

## Scope and limitations

- The supplied case study is simulation-based. It relies on the Webots world and on robot device names and scene nodes expected by the controller.
- The repository includes training and runtime code, but not pretrained PPO policies or anomaly-classifier weights. The workflows below explain how to generate these artifacts locally.
- Anomaly injection is configurable for demonstrations; normal deployment does not require fault-injection settings. The anomaly detectors operate on sensor observations and do not receive fault labels during runtime.
- The task reward, actions, person-detection behavior, and sensor preprocessing are specific to the supplied navigation scenario. Other robots or tasks can reuse the temporal selection architecture with an appropriate environment adapter.
- External robot models, simulation assets, and pretrained third-party models may have additional license requirements.

## Relationship to the research repository

This INTO-CPS case-study presentation is based on the same training and control implementation as the [original research repository](https://github.com/Pavlos-Tosidis/temporal-policy-selection). The Python controller entry points and commands below are intentionally unchanged.

## Acknowledgment

This work was funded by the European Union’s Horizon Europe research and innovation programme under grant agreement No. 101133807 (RoboSAPIENS).

## Citation

For this implementation, use the software citation in [`CITATION.cff`](CITATION.cff). The associated manuscript has been submitted to *Applied Soft Computing* and does not yet have a published journal citation or DOI.

## License

The original implementation is released under the [MIT License](LICENSE), with its copyright notice preserved. The repository's simulation assets, third-party models, and external software retain their respective licenses. See also [`NOTICE`](NOTICE).
