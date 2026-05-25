# Project Architecture Guide

## Project Summary

LeRobot is a Python 3.12+ robotics and robot-learning library built around PyTorch, Hugging Face Hub, draccus configuration, Gymnasium environments, and uv-managed dependencies. It provides:

- Real-robot control abstractions for robots, motors, cameras, and teleoperators.
- A standardized `LeRobotDataset` format for episode-aware robotics data with Parquet metadata/data and image or MP4 visual observations.
- Policy implementations for imitation learning, reinforcement learning, and vision-language-action models.
- CLI tools for recording, replaying, training, evaluation, rollout, calibration, dataset editing, and visualization.
- Documentation, examples, Docker images, CI workflows, and test artifacts for both software-only and hardware-adjacent workflows.

The main package lives under `src/lerobot/`. The public user story is: configure hardware or envs with dataclass configs, collect or load a `LeRobotDataset`, train or load a `PreTrainedPolicy`, pass observations/actions through processor pipelines, then evaluate or deploy through scripts.

## Top-Level Map

```text
.
├── src/lerobot/              # Main Python package
├── tests/                    # Pytest suite, mocks, fixtures, and binary artifacts
├── docs/source/              # Hugging Face documentation source pages
├── examples/                 # Tutorial and workflow scripts
├── docker/                   # User, internal, and benchmark Dockerfiles
├── scripts/ci/               # Small CI parsing/extraction helpers
├── media/readme/             # README images and demo media
├── .github/workflows/        # CI, docs, release, benchmark, and security workflows
├── pyproject.toml            # Package metadata, dependencies, scripts, and tool config
├── uv.lock                   # Locked dependency graph
├── Makefile                  # Docker targets and end-to-end test targets
├── README.md                 # User-facing project overview
├── AGENTS.md                 # Contributor/agent instructions
├── AGENT_GUIDE.md            # Copy-paste helper guide for AI agents and users
└── setup.py                  # Compatibility build entry point
```

## Key Workflows

### Install and Develop

`pyproject.toml` is the source of truth for dependencies and CLI entry points. The repo expects uv:

```bash
uv sync --locked
uv sync --locked --extra test --extra dev
uv sync --locked --extra all
```

Optional extras are feature-scoped: datasets, training, hardware, visualization, motors, robots, policies, simulation envs, async inference, PEFT, tests, and development tools. Many hardware, policy, and simulator dependencies are optional, so imports for those areas are often lazy or guarded.

### Record Data

`lerobot-record` maps to `src/lerobot/scripts/lerobot_record.py`. It constructs robot and teleoperator configs, creates/resumes a `LeRobotDataset`, runs a control loop, sends processed teleop actions to the robot, saves observations/actions, and optionally logs visualizations through Rerun.

Core flow:

```text
Robot.get_observation()
  -> robot observation processor
  -> teleoperator action
  -> teleop action processor
  -> robot action processor
  -> Robot.send_action()
  -> dataset frame write / visualization
```

### Train

`lerobot-train` maps to `src/lerobot/scripts/lerobot_train.py`. It parses `TrainPipelineConfig`, builds dataset, policy, processors, optimizer/scheduler, and optional evaluation environments. It uses Accelerate for device/distributed setup, runs `update_policy()`, logs metrics, saves checkpoints, and can push models to the Hub.

### Evaluate

`lerobot-eval` maps to `src/lerobot/scripts/lerobot_eval.py`. It loads a policy, creates vectorized Gymnasium envs, applies env and policy processors, runs batched rollouts, computes success/reward metrics, and can save videos.

### Rollout / Deploy

`lerobot-rollout` maps to `src/lerobot/scripts/lerobot_rollout.py`. It is the policy-driven real-robot rollout path, separate from teleoperation-only recording.

### Extend

Common extension paths:

- New policy: add `src/lerobot/policies/<name>/configuration_<name>.py`, `modeling_<name>.py`, optionally `processor_<name>.py`, register the config in `PreTrainedConfig`, and update `src/lerobot/policies/factory.py`.
- New robot: subclass `Robot` in `src/lerobot/robots/robot.py`, add a `RobotConfig` subclass in the robot folder, expose it through `robots/__init__.py`/factory helpers, and add tests/mocks where possible.
- New teleoperator: subclass `Teleoperator`, add a config under `src/lerobot/teleoperators/`, and wire it into CLI imports.
- New processor: subclass `ProcessorStep`, register it with `ProcessorStepRegistry.register()`, implement feature transformation and state serialization if needed.
- New env: subclass `EnvConfig`, register with `@EnvConfig.register_subclass(...)`, define `gym_kwargs`, feature maps, and custom processors if required.

## Directory and File Responsibilities

### Main Package: `src/lerobot/`

| Path | Responsibility |
| --- | --- |
| `__init__.py`, `__version__.py` | Package initialization and version metadata. |
| `types.py` | Shared aliases and typed dictionaries for robot, env, policy, and transition data. |
| `configs/` | Draccus dataclass configs for training, eval, dataset recording, policies, rewards, videos, recipes, parsing, and feature types. |
| `scripts/` | CLI implementations exposed by `[project.scripts]` in `pyproject.toml`. |
| `policies/` | Policy base class, factories, and per-policy implementations. |
| `processor/` | Composable data-processing pipeline used between datasets, policies, envs, robots, and teleoperators. |
| `datasets/` | `LeRobotDataset`, metadata, readers/writers, stats, video/image I/O, samplers, aggregation, dataset tools, and streaming support. |
| `envs/` | Gymnasium environment configs, factories, Hub env loading, and benchmark adapters. |
| `robots/` | Hardware robot base class and concrete follower robot implementations. |
| `teleoperators/` | Teleoperation base class and leader/input-device implementations. |
| `motors/` | Motor bus abstraction, calibration helpers, motor encoding, and vendor-specific motor drivers. |
| `cameras/` | Camera base/config classes and OpenCV, RealSense, Reachy2, and ZMQ camera backends. |
| `rollout/` | Real-robot rollout orchestration, context, ring buffers, robot wrappers, inference adapters, and rollout strategies. |
| `rl/` | Reinforcement-learning actors, buffers, learner services, trainer, algorithms, and data sources. |
| `rewards/` | Reward-model base/factory plus classifier and SARM reward components. |
| `async_inference/` | gRPC-based policy server, robot client, configs, constants, and helper utilities. |
| `optim/` | Optimizer and scheduler config/factory implementations. |
| `model/` | Shared model-level utilities, currently including kinematics. |
| `common/` | Shared training/control/W&B utilities used by scripts. |
| `transforms/` | Image/data transforms used by dataset and policy flows. |
| `transport/` | gRPC protobuf definitions, generated Python bindings, and transport helpers. |
| `templates/` | Model card templates for policies and reward models. |
| `utils/` | Shared utilities for device handling, Hub I/O, logging, random seeds, visualization, rotations, sample weighting, process helpers, and constants. |
| `data_processing/` | Dataset-processing helpers, currently including SARM annotation support. |

### Configuration: `src/lerobot/configs/`

| File | Responsibility |
| --- | --- |
| `train.py` | `TrainPipelineConfig`, training validation, resume/pretrained config loading, output directory logic, optimizer/scheduler preset handling, and Hub serialization. |
| `eval.py` | Evaluation pipeline config. |
| `dataset.py` | Dataset recording/loading config dataclasses. |
| `default.py` | Common config blocks such as dataset, eval, WandB, and PEFT settings. |
| `policies.py` | `PreTrainedConfig` base and policy config registry integration. |
| `rewards.py` | Reward-model config registry. |
| `recipe.py` | Recipe config support. |
| `parser.py` | Draccus CLI/config parsing helpers, pretrained path handling, and override extraction. |
| `types.py` | Feature and policy feature definitions used across datasets, policies, and processors. |
| `video.py` | Video encoder configuration and defaults. |

### Policies: `src/lerobot/policies/`

| Path | Responsibility |
| --- | --- |
| `pretrained.py` | `PreTrainedPolicy`, the abstract `nn.Module` + Hub mixin base for save/load, safetensors, action selection, reset, and training forward contracts. |
| `factory.py` | Policy class/config factory, lazy imports, policy creation, and default pre/post processor construction. |
| `utils.py` | Policy loading/validation helpers. |
| `pi_gemma.py` | Shared Pi/Gemma-related helpers. |
| `act/` | ACT policy config, model, and processor. |
| `diffusion/` | Diffusion policy config, model, and processor. |
| `tdmpc/` | TD-MPC policy config, model, and processor. |
| `vqbet/` | VQ-BeT policy config, model, processor, and utilities. |
| `gaussian_actor/` | Gaussian actor policy config, model, and processor. |
| `multi_task_dit/` | Multi-task DiT policy config, model, and processor. |
| `smolvla/` | SmolVLA config, model, VLM-with-expert code, and processor. |
| `pi0/`, `pi05/`, `pi0_fast/` | Pi-family policy configs, models, and processors. |
| `groot/` | GR00T policy, action head modules, Eagle2 Hugging Face model code, and utilities. |
| `xvla/` | XVLA policy, Florence2/action-hub support, soft transformer, config, processor, and utilities. |
| `wall_x/` | Wall-X policy, Qwen model code, constants, config, processor, and utilities. |
| `eo1/` | EO1 policy config, model, and processor. |
| `rtc/` | Real-time control helpers: RTC model/config, action queue, relative action logic, latency/debug tools, and interpolation. |

Each normal policy subdirectory follows a recognizable pattern: `configuration_*.py`, `modeling_*.py`, optional `processor_*.py`, and `__init__.py`.

### Processor Pipeline: `src/lerobot/processor/`

| File | Responsibility |
| --- | --- |
| `pipeline.py` | `ProcessorStepRegistry`, `ProcessorStep`, `DataProcessorPipeline`, migration errors, Hub serialization, state loading/saving, and feature transformation plumbing. |
| `factory.py` | Default processor creation and policy/robot processor helpers. |
| `converters.py` | Batch/transition conversion helpers. |
| `batch_processor.py` | Batch-level processing steps. |
| `observation_processor.py` | Observation-specific processing. |
| `delta_action_processor.py`, `relative_action_processor.py` | Action representation transforms. |
| `normalize_processor.py` | Normalization/unnormalization processors. |
| `device_processor.py` | Device transfer steps. |
| `rename_processor.py` | Feature/key renaming. |
| `tokenizer_processor.py`, `render_messages_processor.py`, `newline_task_processor.py` | Language/message/tokenization-related processors. |
| `env_processor.py`, `gym_action_processor.py`, `hil_processor.py` | Env/HIL/Gym-specific transforms. |
| `policy_robot_bridge.py` | Bridge logic between policy outputs and robot action/observation conventions. |
| `migrate_policy_normalization.py` | Migration support for older policy normalization formats. |

### Datasets: `src/lerobot/datasets/`

| File | Responsibility |
| --- | --- |
| `lerobot_dataset.py` | Main `LeRobotDataset`; handles local/Hub loading, episode filtering, metadata, video decoding, image transforms, create/resume/write modes, and Hub upload support. |
| `dataset_metadata.py` | Dataset versioning, metadata discovery/loading, episode/task/stats metadata. |
| `dataset_reader.py` | Read path for Parquet rows, videos/images, delta timestamps, and active dataset materialization. |
| `dataset_writer.py` | Write path for recorded episodes and dataset structure. |
| `image_writer.py` | Image frame writing and background writer management. |
| `video_utils.py`, `pyav_utils.py` | Video encoding/decoding backends and streaming encoding support. |
| `compute_stats.py` | Dataset statistics computation. |
| `sampler.py` | Episode-aware sampling. |
| `multi_dataset.py` | Multi-dataset utilities/placeholders. |
| `streaming_dataset.py` | Streaming dataset support. |
| `aggregate.py` | Dataset aggregation/merging. |
| `dataset_tools.py` | Dataset edit/split/merge helpers used by tools. |
| `factory.py` | Dataset factory used by training. |
| `feature_utils.py`, `pipeline_features.py` | Feature schema helpers. |
| `language.py`, `language_render.py` | Task/language feature handling and rendering. |
| `io_utils.py`, `utils.py` | Dataset file, Hub, version, and utility helpers. |
| `card_template.md` | Dataset card template. |

The on-disk dataset shape is described directly in `LeRobotDataset`: `data/` chunked Parquet rows, `meta/` JSON/Parquet metadata, and optional `videos/` media chunks.

### Environments: `src/lerobot/envs/`

| File | Responsibility |
| --- | --- |
| `configs.py` | `EnvConfig`, `HubEnvConfig`, built-in env config subclasses, Gymnasium vector env creation, feature maps, and env-specific processors. |
| `factory.py` | `make_env_config()`, `make_env()`, Hub env loading with `trust_remote_code`, and env pre/post processor selection. |
| `utils.py` | Hub env download/import helpers and env validation/preprocessing utilities. |
| `libero.py`, `metaworld.py`, `robocasa.py`, `robomme.py`, `robotwin.py`, `vlabench.py` | Benchmark-specific environment adapters/configs. |
| `metaworld_config.json` | MetaWorld task/config data packaged with the library. |

### Hardware Abstractions

| Path | Responsibility |
| --- | --- |
| `robots/robot.py` | Abstract `Robot` contract: observation/action feature schemas, connect/disconnect, calibration, configure, `get_observation()`, and `send_action()`. |
| `robots/config.py` | Robot config registry/base. |
| `robots/utils.py` | Shared robot helpers. |
| `robots/*_follower/`, `robots/reachy2/`, `robots/unitree_g1/`, etc. | Concrete robot implementations and configs. |
| `teleoperators/teleoperator.py` | Abstract teleoperator contract. |
| `teleoperators/config.py` | Teleoperator config registry/base. |
| `teleoperators/*_leader/`, `teleoperators/keyboard/`, `teleoperators/gamepad/`, `teleoperators/phone/`, etc. | Concrete teleoperation inputs and configs. |
| `motors/motors_bus.py` | Shared motor bus abstraction. |
| `motors/encoding_utils.py` | Motor unit/encoding conversion helpers. |
| `motors/calibration_gui.py` | Calibration UI helper. |
| `motors/dynamixel/`, `motors/feetech/`, `motors/damiao/`, `motors/robstride/` | Vendor-specific motor drivers and tables. |
| `cameras/camera.py`, `cameras/configs.py` | Camera base class and config registry. |
| `cameras/opencv/`, `cameras/realsense/`, `cameras/reachy2_camera/`, `cameras/zmq/` | Concrete camera backends. |

### CLI Scripts: `src/lerobot/scripts/`

| Script | CLI | Responsibility |
| --- | --- | --- |
| `lerobot_train.py` | `lerobot-train` | Train policies/reward models from datasets with checkpointing/eval/Hub support. |
| `lerobot_eval.py` | `lerobot-eval` | Evaluate policies in vectorized environments. |
| `lerobot_record.py` | `lerobot-record` | Teleoperation data collection into `LeRobotDataset`. |
| `lerobot_rollout.py` | `lerobot-rollout` | Policy-driven real-robot rollout/deployment. |
| `lerobot_replay.py` | `lerobot-replay` | Replay dataset episodes on a robot. |
| `lerobot_teleoperate.py` | `lerobot-teleoperate` | Teleoperate a robot without recording. |
| `lerobot_calibrate.py` | `lerobot-calibrate` | Calibrate robot or teleoperator hardware. |
| `lerobot_setup_motors.py` | `lerobot-setup-motors` | Configure motor IDs/baudrate. |
| `lerobot_setup_can.py` | `lerobot-setup-can` | CAN setup helper. |
| `lerobot_find_port.py` | `lerobot-find-port` | Locate serial ports for hardware. |
| `lerobot_find_cameras.py` | `lerobot-find-cameras` | Discover camera devices. |
| `lerobot_find_joint_limits.py` | `lerobot-find-joint-limits` | Hardware joint-limit discovery. |
| `lerobot_dataset_viz.py` | `lerobot-dataset-viz` | Dataset visualization. |
| `lerobot_imgtransform_viz.py` | `lerobot-imgtransform-viz` | Image transform visualization. |
| `lerobot_edit_dataset.py` | `lerobot-edit-dataset` | Dataset editing operations. |
| `lerobot_train_tokenizer.py` | `lerobot-train-tokenizer` | Tokenizer training helper. |
| `lerobot_info.py` | `lerobot-info` | Installation/package info command. |
| `convert_dataset_v21_to_v30.py` | direct script | Dataset format migration. |
| `augment_dataset_quantile_stats.py` | direct script | Dataset quantile/stat augmentation. |

### Tests: `tests/`

| Path | Responsibility |
| --- | --- |
| `conftest.py`, `utils.py` | Shared pytest fixtures, skip helpers, and utility assertions. |
| `configs/` | Config parsing/default/plugin tests. |
| `datasets/` | Dataset loading, metadata, writer/reader, stats, video, transforms, streaming, language, and tool tests. |
| `processor/` | Processor pipeline, migration, converters, policy/robot bridge, per-policy processors, and normalization tests. |
| `policies/` | Policy behavior/regression tests by model family. |
| `envs/` | Environment dispatch and benchmark adapter tests. |
| `training/` | Training workflow and multi-GPU/visual validation tests. |
| `robots/`, `motors/`, `cameras/`, `teleoperators/` | Hardware abstraction tests, generally using mocks or skip decorators for unavailable hardware. |
| `rl/` | RL actors, learner, queues, SAC, trainer, and data mixer tests. |
| `rewards/` | Reward model, classifier, and SARM tests. |
| `async_inference/`, `transport/` | Async/gRPC client-server and transport utility tests. |
| `fixtures/` | Dataset factories, constants, Hub helpers, files, and optimizer fixtures. |
| `mocks/` | Mock robots, teleoperators, motors, serial patches, and vendor motor drivers. |
| `artifacts/` | Small binary fixtures: safetensors, images, videos, and dataset frames. |
| `scripts/`, `utils/` | Script parsing and utility function tests. |

### Docs, Examples, Docker, CI

| Path | Responsibility |
| --- | --- |
| `docs/source/` | User documentation for installation, hardware, policies, datasets, processors, async inference, benchmarks, and tutorials. `_toctree.yml` controls docs navigation. |
| `examples/dataset/` | Dataset loading, transforms, tools, progress videos, and RA-BC/SLURM examples. |
| `examples/training/` | Training examples including streaming. |
| `examples/tutorial/` | Policy and RL tutorials for ACT, Diffusion, Pi0, SmolVLA, async inference, and HIL-SERL. |
| `examples/lekiwi/`, `examples/phone_to_so100/`, `examples/so100_to_so100_EE/`, `examples/omx/` | Hardware-specific teleoperate/record/replay/evaluate/rollout examples. |
| `examples/port_datasets/` | Dataset porting and sharded upload helpers. |
| `docker/` | User/internal Dockerfiles and benchmark-specific images for LIBERO, MetaWorld, RoboCasa, RoboMME, RoboCerebra, RobotWin, and VLABench. |
| `.github/workflows/quality.yml` | Pre-commit quality checks. |
| `.github/workflows/fast_tests.yml` | Fast PR test workflow. |
| `.github/workflows/full_tests.yml` | Broader tests and end-to-end coverage. |
| `.github/workflows/latest_deps_tests.yml` | Daily/latest dependency testing. |
| `.github/workflows/security.yml` | Security scanning. |
| `.github/workflows/release.yml` | Release publishing. |
| `.github/workflows/documentation*.yml` | Documentation build/upload workflows. |
| `.github/workflows/benchmark_tests.yml`, `docker_publish.yml` | Benchmark and Docker publish automation. |

## Core Modules / Core Assets

### `TrainPipelineConfig`

`src/lerobot/configs/train.py` owns the top-level train config. It:

- Resolves `--policy.path` and `--reward_model.path` pretrained configs.
- Handles local checkpoint resume through `config_path`.
- Builds default output directories and job names.
- Validates policy/reward model presence.
- Applies optimizer/scheduler presets from the active trainable config.
- Serializes `train_config.json` into checkpoints and Hub uploads.

### `PreTrainedPolicy`

`src/lerobot/policies/pretrained.py` is the policy base contract. Subclasses must define `config_class` and `name`, implement optimizer params, `reset()`, `forward()`, `predict_action_chunk()`, and `select_action()`. Save/load uses config files plus `model.safetensors`, with Hub download support.

### Policy Factory

`src/lerobot/policies/factory.py` maps policy names to config/model classes. It deliberately uses lazy imports for modeling classes to avoid importing heavyweight optional dependencies until a policy is requested. It also creates policy pre/post processor pipelines and reconnects relative/absolute action processor state after deserialization.

### `LeRobotDataset`

`src/lerobot/datasets/lerobot_dataset.py` is the dataset entry point. It supports loading from local disk or Hub, filtering episodes, decoding videos/images, applying image transforms, downloading missing data, creating new datasets, resuming existing datasets, and writing/uploading recordings.

### `DataProcessorPipeline`

`src/lerobot/processor/pipeline.py` provides the composable transformation model used throughout the project. Processor steps transform transitions and feature schemas, can carry tensor state, can be serialized to/from Hub-compatible files, and are registered by name for reconstruction.

### `EnvConfig`

`src/lerobot/envs/configs.py` defines the environment registry pattern. Built-in env configs specify Gym IDs, feature maps, FPS, observation/action schemas, and vectorized env creation. `HubEnvConfig` supports remote env definitions with explicit `trust_remote_code`.

### `Robot`

`src/lerobot/robots/robot.py` defines the physical robot interface. Concrete robots expose action/observation features, connection state, calibration, configuration, observation reads, action writes, and cleanup.

## Configuration, Build, and Runtime Entry Points

| File | Responsibility |
| --- | --- |
| `pyproject.toml` | Build metadata, package dependencies, optional extras, CLI scripts, uv source/index config, setuptools package data, Ruff, Bandit, typos, and mypy settings. |
| `uv.lock` | Locked dependency versions. |
| `requirements.in` | Requirement input compatibility file. |
| `docs-requirements.txt` | Documentation build dependencies. |
| `requirements-ubuntu.txt`, `requirements-macos.txt` | OS-level setup references. |
| `Makefile` | Docker image targets and end-to-end train/eval test recipes for ACT, Diffusion, TD-MPC, and SmolVLA. |
| `MANIFEST.in` | Source distribution inclusion rules. |
| `setup.py` | Compatibility setup entry point for setuptools. |
| `LICENSE`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `AI_POLICY.md` | Project policy, security, contribution, and governance documents. |
| `README.md` | Project overview and user-facing quickstart. |
| `AGENTS.md` | Development guidance for AI agents in this repository. |
| `AGENT_GUIDE.md` | Practical user-facing command guide for agents helping with LeRobot workflows. |

Runtime CLIs are installed from `[project.scripts]` in `pyproject.toml`, all pointing to `src/lerobot/scripts/*.py`.

## Tests and Quality Checks

Primary commands verified from repository files:

```bash
uv run pytest tests -svv --maxfail=10
DEVICE=cuda make test-end-to-end
pre-commit run --all-files
```

Quality/tooling configuration:

- Ruff targets Python 3.12, line length 110, double quotes, isort first-party package `lerobot`.
- Bandit is configured but skips tests, benchmarks, and selected rule IDs.
- Mypy is gradual: `lerobot.*` is mostly ignored, while `envs`, `configs`, `optim`, `model`, `cameras`, `motors`, and `transport` are checked more strictly.
- `tests/artifacts/**/*.safetensors` and protobuf-generated files are excluded from Ruff.

E2E tests in the `Makefile` create short train/eval runs under `tests/outputs/` for representative policies and envs.

## Generated, Repetitive, or Low-Signal Files

- `uv.lock` is generated by uv and should not be hand-edited.
- `src/lerobot/transport/services_pb2.py` and `services_pb2_grpc.py` are generated from `services.proto`.
- `tests/artifacts/` contains binary fixtures (`.safetensors`, `.mp4`, `.png`, `.bag`) used by tests; inspect by role/path rather than treating them as source.
- `media/readme/` contains README visual assets.
- Policy subdirectories repeat the `configuration_*`, `modeling_*`, `processor_*` pattern; describe the family unless a specific policy is under change.
- Robot/teleoperator subdirectories repeat the `config_*` plus implementation pattern for each hardware family.
- Docs pages in `docs/source/` are numerous topic pages; `_toctree.yml` is the navigation anchor.
- `src/lerobot.egg-info/` and `__pycache__/` are build/runtime artifacts present in this checkout and are not source of truth.

## Common Change Paths

| Goal | Start Here | Also Check |
| --- | --- | --- |
| Change training behavior | `src/lerobot/scripts/lerobot_train.py`, `src/lerobot/configs/train.py` | `src/lerobot/common/train_utils.py`, `tests/training/`, `Makefile` |
| Add or modify a policy | `src/lerobot/policies/<policy>/`, `src/lerobot/policies/factory.py` | `src/lerobot/configs/policies.py`, `tests/policies/`, policy docs |
| Change dataset loading/writing | `src/lerobot/datasets/lerobot_dataset.py` | `dataset_reader.py`, `dataset_writer.py`, `video_utils.py`, `tests/datasets/` |
| Add dataset tooling | `src/lerobot/datasets/dataset_tools.py`, `src/lerobot/scripts/lerobot_edit_dataset.py` | `tests/datasets/test_dataset_tools.py`, docs/source/using_dataset_tools.mdx |
| Add a processor | `src/lerobot/processor/pipeline.py`, relevant processor file | `tests/processor/`, docs/source/implement_your_own_processor.mdx |
| Add an environment | `src/lerobot/envs/configs.py`, `src/lerobot/envs/factory.py` | `tests/envs/`, docs/source/envhub.mdx, optional deps in `pyproject.toml` |
| Add robot hardware | `src/lerobot/robots/robot.py`, `src/lerobot/robots/<name>/` | `src/lerobot/motors/`, `src/lerobot/cameras/`, `tests/robots/`, hardware docs |
| Add teleoperation hardware | `src/lerobot/teleoperators/teleoperator.py`, `src/lerobot/teleoperators/<name>/` | `tests/teleoperators/`, recording/teleop scripts |
| Modify camera support | `src/lerobot/cameras/` | `tests/cameras/`, optional deps in `pyproject.toml` |
| Modify motor drivers | `src/lerobot/motors/` | `tests/motors/`, setup/calibration scripts |
| Change docs navigation | `docs/source/_toctree.yml` | Relevant `.mdx` page and policy README docs |
| Change CLI availability | `[project.scripts]` in `pyproject.toml` | Matching file in `src/lerobot/scripts/` |
| Change CI/test behavior | `.github/workflows/`, `Makefile` | `pyproject.toml` tool configs |

## Open Questions or Caveats

- Some simulator integrations are intentionally not normal PyPI extras because their dependency graphs or distribution models conflict with the base project; see comments in `pyproject.toml` and docs pages for exact install guidance.
- Hardware-facing tests may skip depending on platform, connected devices, or optional dependencies.
- Binary fixtures and generated protobuf files were classified by filename/path and references, not by deep binary inspection.
- This guide was created from repository inspection, not from executing the full test suite.
