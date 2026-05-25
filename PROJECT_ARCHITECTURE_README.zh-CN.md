# 项目架构指南

## 项目概览

LeRobot 是一个面向真实机器人和机器人学习的 Python 3.12+ 库，核心技术栈包括 PyTorch、Hugging Face Hub、draccus 配置系统、Gymnasium 环境，以及 uv 依赖管理。它提供：

- 面向机器人、电机、相机和遥操作设备的真实硬件控制抽象。
- 标准化的 `LeRobotDataset` 数据集格式，支持按 episode 组织的机器人数据、Parquet 元数据/数据，以及图像或 MP4 视觉观测。
- 覆盖模仿学习、强化学习、视觉-语言-动作模型的策略实现。
- 用于录制、回放、训练、评估、rollout、校准、数据集编辑和可视化的 CLI 工具。
- 文档、示例、Docker 镜像、CI 工作流，以及软件和硬件相关测试所需的测试资产。

主包位于 `src/lerobot/`。典型使用链路是：通过 dataclass 配置硬件或环境，采集或加载 `LeRobotDataset`，训练或加载 `PreTrainedPolicy`，通过 processor pipeline 在数据集、策略、环境和机器人之间转换观测/动作，最后用脚本完成评估或部署。

## 顶层结构

```text
.
├── src/lerobot/              # 主 Python 包
├── tests/                    # Pytest 测试、mock、fixture 和二进制测试资产
├── docs/source/              # Hugging Face 文档源文件
├── examples/                 # 教程和工作流脚本
├── docker/                   # 用户、内部和 benchmark Dockerfile
├── scripts/ci/               # CI 解析/提取辅助脚本
├── media/readme/             # README 图片和演示媒体
├── .github/workflows/        # CI、文档、发布、benchmark、安全扫描工作流
├── pyproject.toml            # 包元数据、依赖、CLI 脚本和工具配置
├── uv.lock                   # 锁定的依赖图
├── Makefile                  # Docker 目标和端到端测试目标
├── README.md                 # 面向用户的项目概览
├── AGENTS.md                 # 贡献者/AI agent 工作说明
├── AGENT_GUIDE.md            # 面向用户和 AI agent 的可复制命令指南
└── setup.py                  # setuptools 兼容构建入口
```

## 关键工作流

### 安装和开发

`pyproject.toml` 是依赖和 CLI 入口的事实来源。仓库默认使用 uv：

```bash
uv sync --locked
uv sync --locked --extra test --extra dev
uv sync --locked --extra all
```

可选 extras 按功能划分：datasets、training、hardware、visualization、motors、robots、policies、simulation envs、async inference、PEFT、tests 和 development tools。很多硬件、策略和模拟器依赖都是可选的，因此相关模块通常采用延迟导入或导入保护。

### 录制数据

`lerobot-record` 对应 `src/lerobot/scripts/lerobot_record.py`。它会构造机器人和遥操作设备配置，创建或恢复 `LeRobotDataset`，运行控制循环，将处理后的遥操作动作发送给机器人，保存观测/动作，并可选通过 Rerun 记录可视化数据。

核心流程：

```text
Robot.get_observation()
  -> robot observation processor
  -> teleoperator action
  -> teleop action processor
  -> robot action processor
  -> Robot.send_action()
  -> dataset frame write / visualization
```

### 训练

`lerobot-train` 对应 `src/lerobot/scripts/lerobot_train.py`。它解析 `TrainPipelineConfig`，构建数据集、策略、processor、优化器/调度器，以及可选的评估环境。训练脚本使用 Accelerate 做设备和分布式设置，调用 `update_policy()`，记录指标，保存 checkpoint，并可将模型推送到 Hugging Face Hub。

### 评估

`lerobot-eval` 对应 `src/lerobot/scripts/lerobot_eval.py`。它加载策略，创建向量化 Gymnasium 环境，应用环境和策略 processor，运行批量 rollout，计算 success/reward 指标，并可保存视频。

### Rollout / 部署

`lerobot-rollout` 对应 `src/lerobot/scripts/lerobot_rollout.py`。这是基于策略的真实机器人 rollout/部署路径，和只做遥操作录制的 `lerobot-record` 分开。

### 扩展

常见扩展路径：

- 新策略：新增 `src/lerobot/policies/<name>/configuration_<name>.py`、`modeling_<name>.py`，可选新增 `processor_<name>.py`；在 `PreTrainedConfig` 中注册配置，并更新 `src/lerobot/policies/factory.py`。
- 新机器人：继承 `src/lerobot/robots/robot.py` 中的 `Robot`，在机器人目录中新增 `RobotConfig` 子类，通过 `robots/__init__.py`/factory helper 暴露，并尽量添加测试或 mock。
- 新遥操作设备：继承 `Teleoperator`，在 `src/lerobot/teleoperators/` 下新增配置，并接入 CLI 导入。
- 新 processor：继承 `ProcessorStep`，用 `ProcessorStepRegistry.register()` 注册，实现 feature transformation，如有状态还要实现状态序列化。
- 新环境：继承 `EnvConfig`，用 `@EnvConfig.register_subclass(...)` 注册，定义 `gym_kwargs`、feature map，以及需要的自定义 processor。

## 目录和文件职责

### 主包：`src/lerobot/`

| 路径 | 职责 |
| --- | --- |
| `__init__.py`, `__version__.py` | 包初始化和版本元数据。 |
| `types.py` | 机器人、环境、策略和 transition 数据共用的类型别名和 typed dict。 |
| `configs/` | 基于 draccus 的 dataclass 配置：训练、评估、数据集录制、策略、奖励、视频、recipe、解析和 feature 类型。 |
| `scripts/` | `pyproject.toml` 中 `[project.scripts]` 暴露的 CLI 实现。 |
| `policies/` | 策略基类、factory，以及各策略实现。 |
| `processor/` | 可组合数据处理流水线，用于数据集、策略、环境、机器人和遥操作设备之间。 |
| `datasets/` | `LeRobotDataset`、元数据、reader/writer、统计、视频/图像 I/O、sampler、聚合、数据集工具和 streaming 支持。 |
| `envs/` | Gymnasium 环境配置、factory、Hub 环境加载和 benchmark 适配器。 |
| `robots/` | 硬件机器人基类和具体 follower 机器人实现。 |
| `teleoperators/` | 遥操作基类，以及 leader/input device 实现。 |
| `motors/` | motor bus 抽象、校准辅助、电机编码和厂商电机驱动。 |
| `cameras/` | 相机基类/配置，以及 OpenCV、RealSense、Reachy2、ZMQ 相机后端。 |
| `rollout/` | 真实机器人 rollout 编排、上下文、ring buffer、机器人 wrapper、推理适配器和 rollout 策略。 |
| `rl/` | 强化学习 actor、buffer、learner service、trainer、算法和数据源。 |
| `rewards/` | 奖励模型基类/factory，以及 classifier 和 SARM 奖励组件。 |
| `async_inference/` | 基于 gRPC 的 policy server、robot client、配置、常量和辅助函数。 |
| `optim/` | 优化器和 scheduler 的配置/factory。 |
| `model/` | 共享模型级工具，目前包括 kinematics。 |
| `common/` | 训练、控制、W&B 等脚本共用工具。 |
| `transforms/` | 数据集和策略流程使用的图像/数据 transform。 |
| `transport/` | gRPC protobuf 定义、生成的 Python 绑定和 transport helper。 |
| `templates/` | 策略和奖励模型的 model card 模板。 |
| `utils/` | 设备、Hub I/O、日志、随机种子、可视化、旋转、sample weighting、进程辅助和常量等通用工具。 |
| `data_processing/` | 数据集处理辅助代码，目前包括 SARM annotation 支持。 |

### 配置：`src/lerobot/configs/`

| 文件 | 职责 |
| --- | --- |
| `train.py` | `TrainPipelineConfig`、训练配置校验、resume/pretrained 配置加载、输出目录逻辑、优化器/scheduler preset 和 Hub 序列化。 |
| `eval.py` | 评估 pipeline 配置。 |
| `dataset.py` | 数据集录制/加载相关配置 dataclass。 |
| `default.py` | 常用配置块，如 dataset、eval、WandB、PEFT 设置。 |
| `policies.py` | `PreTrainedConfig` 基类和策略配置注册。 |
| `rewards.py` | 奖励模型配置注册。 |
| `recipe.py` | Recipe 配置支持。 |
| `parser.py` | Draccus CLI/config 解析辅助、pretrained path 处理和 override 提取。 |
| `types.py` | 数据集、策略和 processor 共用的 feature 和 policy feature 定义。 |
| `video.py` | 视频编码器配置和默认值。 |

### 策略：`src/lerobot/policies/`

| 路径 | 职责 |
| --- | --- |
| `pretrained.py` | `PreTrainedPolicy`，即抽象 `nn.Module` + Hub mixin 基类，负责 save/load、safetensors、动作选择、reset 和训练 forward 约定。 |
| `factory.py` | 策略类/配置 factory、延迟导入、策略创建、默认 pre/post processor 构造。 |
| `utils.py` | 策略加载和校验辅助函数。 |
| `pi_gemma.py` | Pi/Gemma 相关共享辅助代码。 |
| `act/` | ACT 策略配置、模型和 processor。 |
| `diffusion/` | Diffusion 策略配置、模型和 processor。 |
| `tdmpc/` | TD-MPC 策略配置、模型和 processor。 |
| `vqbet/` | VQ-BeT 策略配置、模型、processor 和工具。 |
| `gaussian_actor/` | Gaussian actor 策略配置、模型和 processor。 |
| `multi_task_dit/` | Multi-task DiT 策略配置、模型和 processor。 |
| `smolvla/` | SmolVLA 配置、模型、VLM-with-expert 代码和 processor。 |
| `pi0/`, `pi05/`, `pi0_fast/` | Pi 系列策略配置、模型和 processor。 |
| `groot/` | GR00T 策略、action head 模块、Eagle2 Hugging Face 模型代码和工具。 |
| `xvla/` | XVLA 策略、Florence2/action-hub 支持、soft transformer、配置、processor 和工具。 |
| `wall_x/` | Wall-X 策略、Qwen 模型代码、常量、配置、processor 和工具。 |
| `eo1/` | EO1 策略配置、模型和 processor。 |
| `rtc/` | 实时控制相关辅助：RTC 模型/配置、action queue、relative action、延迟/调试工具和插值。 |

普通策略子目录通常遵循 `configuration_*.py`、`modeling_*.py`、可选 `processor_*.py`、`__init__.py` 的结构。

### Processor Pipeline：`src/lerobot/processor/`

| 文件 | 职责 |
| --- | --- |
| `pipeline.py` | `ProcessorStepRegistry`、`ProcessorStep`、`DataProcessorPipeline`、迁移错误、Hub 序列化、状态加载/保存和 feature transformation 管线。 |
| `factory.py` | 默认 processor 创建，以及 policy/robot processor 辅助函数。 |
| `converters.py` | batch/transition 转换辅助函数。 |
| `batch_processor.py` | batch 级处理步骤。 |
| `observation_processor.py` | observation 相关处理。 |
| `delta_action_processor.py`, `relative_action_processor.py` | 动作表示变换。 |
| `normalize_processor.py` | 归一化/反归一化 processor。 |
| `device_processor.py` | 设备迁移步骤。 |
| `rename_processor.py` | feature/key 重命名。 |
| `tokenizer_processor.py`, `render_messages_processor.py`, `newline_task_processor.py` | 语言、消息渲染和 tokenization 相关 processor。 |
| `env_processor.py`, `gym_action_processor.py`, `hil_processor.py` | 环境、HIL、Gym 相关变换。 |
| `policy_robot_bridge.py` | 策略输出和机器人 action/observation 约定之间的桥接逻辑。 |
| `migrate_policy_normalization.py` | 旧策略归一化格式迁移支持。 |

### 数据集：`src/lerobot/datasets/`

| 文件 | 职责 |
| --- | --- |
| `lerobot_dataset.py` | 主入口 `LeRobotDataset`；处理本地/Hub 加载、episode 过滤、元数据、视频解码、图像 transform、create/resume/write 模式和 Hub 上传。 |
| `dataset_metadata.py` | 数据集版本、元数据发现/加载、episode/task/stats 元数据。 |
| `dataset_reader.py` | Parquet 行、视频/图像、delta timestamps 和 active dataset materialization 的读取路径。 |
| `dataset_writer.py` | 录制 episode 和数据集结构的写入路径。 |
| `image_writer.py` | 图像帧写入和后台 writer 管理。 |
| `video_utils.py`, `pyav_utils.py` | 视频编码/解码后端和 streaming encoding 支持。 |
| `compute_stats.py` | 数据集统计计算。 |
| `sampler.py` | Episode-aware sampling。 |
| `multi_dataset.py` | 多数据集工具/占位支持。 |
| `streaming_dataset.py` | Streaming dataset 支持。 |
| `aggregate.py` | 数据集聚合/合并。 |
| `dataset_tools.py` | 数据集编辑、切分、合并等工具使用的辅助逻辑。 |
| `factory.py` | 训练流程使用的数据集 factory。 |
| `feature_utils.py`, `pipeline_features.py` | Feature schema 辅助函数。 |
| `language.py`, `language_render.py` | 任务/语言 feature 处理和渲染。 |
| `io_utils.py`, `utils.py` | 数据集文件、Hub、版本和通用工具。 |
| `card_template.md` | 数据集 card 模板。 |

磁盘上的数据集结构在 `LeRobotDataset` 中有直接说明：`data/` 存放分块 Parquet 行数据，`meta/` 存放 JSON/Parquet 元数据，可选 `videos/` 存放媒体分块。

### 环境：`src/lerobot/envs/`

| 文件 | 职责 |
| --- | --- |
| `configs.py` | `EnvConfig`、`HubEnvConfig`、内置环境配置子类、Gymnasium vector env 创建、feature map 和环境专用 processor。 |
| `factory.py` | `make_env_config()`、`make_env()`、带 `trust_remote_code` 的 Hub 环境加载，以及 env pre/post processor 选择。 |
| `utils.py` | Hub 环境下载/导入辅助函数，以及环境校验/预处理工具。 |
| `libero.py`, `metaworld.py`, `robocasa.py`, `robomme.py`, `robotwin.py`, `vlabench.py` | benchmark 专用环境适配器/配置。 |
| `metaworld_config.json` | 随包分发的 MetaWorld task/config 数据。 |

### 硬件抽象

| 路径 | 职责 |
| --- | --- |
| `robots/robot.py` | 抽象 `Robot` 合约：observation/action feature schema、connect/disconnect、calibration、configure、`get_observation()` 和 `send_action()`。 |
| `robots/config.py` | Robot 配置注册/基类。 |
| `robots/utils.py` | 共享机器人辅助函数。 |
| `robots/*_follower/`, `robots/reachy2/`, `robots/unitree_g1/`, etc. | 具体机器人实现和配置。 |
| `teleoperators/teleoperator.py` | 抽象 teleoperator 合约。 |
| `teleoperators/config.py` | Teleoperator 配置注册/基类。 |
| `teleoperators/*_leader/`, `teleoperators/keyboard/`, `teleoperators/gamepad/`, `teleoperators/phone/`, etc. | 具体遥操作输入设备和配置。 |
| `motors/motors_bus.py` | 共享 motor bus 抽象。 |
| `motors/encoding_utils.py` | 电机单位/编码转换辅助。 |
| `motors/calibration_gui.py` | 校准 UI 辅助。 |
| `motors/dynamixel/`, `motors/feetech/`, `motors/damiao/`, `motors/robstride/` | 厂商电机驱动和表定义。 |
| `cameras/camera.py`, `cameras/configs.py` | 相机基类和配置注册。 |
| `cameras/opencv/`, `cameras/realsense/`, `cameras/reachy2_camera/`, `cameras/zmq/` | 具体相机后端。 |

### CLI 脚本：`src/lerobot/scripts/`

| 脚本 | CLI | 职责 |
| --- | --- | --- |
| `lerobot_train.py` | `lerobot-train` | 从数据集训练策略/奖励模型，支持 checkpoint、评估和 Hub。 |
| `lerobot_eval.py` | `lerobot-eval` | 在向量化环境中评估策略。 |
| `lerobot_record.py` | `lerobot-record` | 通过遥操作采集数据到 `LeRobotDataset`。 |
| `lerobot_rollout.py` | `lerobot-rollout` | 基于策略的真实机器人 rollout/部署。 |
| `lerobot_replay.py` | `lerobot-replay` | 在机器人上回放数据集 episode。 |
| `lerobot_teleoperate.py` | `lerobot-teleoperate` | 只遥操作机器人，不录制。 |
| `lerobot_calibrate.py` | `lerobot-calibrate` | 校准机器人或遥操作硬件。 |
| `lerobot_setup_motors.py` | `lerobot-setup-motors` | 配置电机 ID/baudrate。 |
| `lerobot_setup_can.py` | `lerobot-setup-can` | CAN 设置辅助。 |
| `lerobot_find_port.py` | `lerobot-find-port` | 查找硬件串口。 |
| `lerobot_find_cameras.py` | `lerobot-find-cameras` | 发现相机设备。 |
| `lerobot_find_joint_limits.py` | `lerobot-find-joint-limits` | 硬件关节限位发现。 |
| `lerobot_dataset_viz.py` | `lerobot-dataset-viz` | 数据集可视化。 |
| `lerobot_imgtransform_viz.py` | `lerobot-imgtransform-viz` | 图像 transform 可视化。 |
| `lerobot_edit_dataset.py` | `lerobot-edit-dataset` | 数据集编辑操作。 |
| `lerobot_train_tokenizer.py` | `lerobot-train-tokenizer` | tokenizer 训练辅助。 |
| `lerobot_info.py` | `lerobot-info` | 安装/包信息命令。 |
| `convert_dataset_v21_to_v30.py` | 直接脚本 | 数据集格式迁移。 |
| `augment_dataset_quantile_stats.py` | 直接脚本 | 数据集 quantile/stat 增强。 |

### 测试：`tests/`

| 路径 | 职责 |
| --- | --- |
| `conftest.py`, `utils.py` | 共享 pytest fixture、skip helper 和断言工具。 |
| `configs/` | 配置解析、默认值和 plugin 测试。 |
| `datasets/` | 数据集加载、元数据、writer/reader、统计、视频、transform、streaming、语言和工具测试。 |
| `processor/` | Processor pipeline、迁移、converter、policy/robot bridge、各策略 processor 和归一化测试。 |
| `policies/` | 按模型家族组织的策略行为/回归测试。 |
| `envs/` | 环境 dispatch 和 benchmark adapter 测试。 |
| `training/` | 训练工作流、多 GPU 和 visual validation 测试。 |
| `robots/`, `motors/`, `cameras/`, `teleoperators/` | 硬件抽象测试，通常使用 mock 或针对不可用硬件的 skip decorator。 |
| `rl/` | RL actor、learner、queue、SAC、trainer 和 data mixer 测试。 |
| `rewards/` | 奖励模型、classifier 和 SARM 测试。 |
| `async_inference/`, `transport/` | async/gRPC client-server 和 transport 工具测试。 |
| `fixtures/` | 数据集 factory、常量、Hub helper、文件和 optimizer fixture。 |
| `mocks/` | Mock 机器人、遥操作设备、电机、serial patch 和厂商电机驱动。 |
| `artifacts/` | 小型二进制 fixture：safetensors、图片、视频和数据集帧。 |
| `scripts/`, `utils/` | 脚本解析和工具函数测试。 |

### 文档、示例、Docker、CI

| 路径 | 职责 |
| --- | --- |
| `docs/source/` | 安装、硬件、策略、数据集、processor、async inference、benchmark 和教程文档。`_toctree.yml` 控制文档导航。 |
| `examples/dataset/` | 数据集加载、transform、工具、progress video 和 RA-BC/SLURM 示例。 |
| `examples/training/` | 训练示例，包括 streaming。 |
| `examples/tutorial/` | ACT、Diffusion、Pi0、SmolVLA、async inference 和 HIL-SERL 的策略/RL 教程。 |
| `examples/lekiwi/`, `examples/phone_to_so100/`, `examples/so100_to_so100_EE/`, `examples/omx/` | 硬件专用 teleoperate/record/replay/evaluate/rollout 示例。 |
| `examples/port_datasets/` | 数据集迁移和分片上传辅助。 |
| `docker/` | 用户/内部 Dockerfile，以及 LIBERO、MetaWorld、RoboCasa、RoboMME、RoboCerebra、RobotWin、VLABench 的 benchmark 镜像。 |
| `.github/workflows/quality.yml` | pre-commit 质量检查。 |
| `.github/workflows/fast_tests.yml` | 快速 PR 测试工作流。 |
| `.github/workflows/full_tests.yml` | 更完整的测试和端到端覆盖。 |
| `.github/workflows/latest_deps_tests.yml` | 每日/latest dependency 测试。 |
| `.github/workflows/security.yml` | 安全扫描。 |
| `.github/workflows/release.yml` | 发布。 |
| `.github/workflows/documentation*.yml` | 文档构建/上传工作流。 |
| `.github/workflows/benchmark_tests.yml`, `docker_publish.yml` | benchmark 和 Docker 发布自动化。 |

## 核心模块 / 核心资产

### `TrainPipelineConfig`

`src/lerobot/configs/train.py` 负责顶层训练配置。它会：

- 解析 `--policy.path` 和 `--reward_model.path` 指向的 pretrained 配置。
- 通过 `config_path` 处理本地 checkpoint resume。
- 构造默认输出目录和 job name。
- 校验必须配置 policy 或 reward model。
- 从当前 trainable config 应用 optimizer/scheduler preset。
- 将 `train_config.json` 序列化到 checkpoint 和 Hub 上传目录中。

### `PreTrainedPolicy`

`src/lerobot/policies/pretrained.py` 是策略基类合约。子类必须定义 `config_class` 和 `name`，并实现 optimizer params、`reset()`、`forward()`、`predict_action_chunk()` 和 `select_action()`。保存/加载使用配置文件加 `model.safetensors`，并支持从 Hugging Face Hub 下载。

### Policy Factory

`src/lerobot/policies/factory.py` 将策略名映射到配置类和模型类。它刻意对 modeling 类使用延迟导入，避免在未请求某个策略时就导入重量级可选依赖。它还负责创建 policy pre/post processor pipeline，并在反序列化后重新连接 relative/absolute action processor 的状态引用。

### `LeRobotDataset`

`src/lerobot/datasets/lerobot_dataset.py` 是数据集入口。它支持从本地磁盘或 Hub 加载、过滤 episode、解码视频/图像、应用图像 transform、下载缺失数据、创建新数据集、恢复已有数据集，以及写入/上传录制结果。

### `DataProcessorPipeline`

`src/lerobot/processor/pipeline.py` 提供贯穿项目的可组合转换模型。Processor step 会转换 transition 和 feature schema，可以携带 tensor 状态，可以序列化为 Hub 兼容文件，并通过名称注册以便重建。

### `EnvConfig`

`src/lerobot/envs/configs.py` 定义环境注册模式。内置环境配置会指定 Gym ID、feature map、FPS、observation/action schema 和向量化环境创建逻辑。`HubEnvConfig` 支持远程环境定义，但需要显式设置 `trust_remote_code`。

### `Robot`

`src/lerobot/robots/robot.py` 定义物理机器人接口。具体机器人需要暴露 action/observation features、连接状态、校准、配置、观测读取、动作写入和清理逻辑。

## 配置、构建和运行入口

| 文件 | 职责 |
| --- | --- |
| `pyproject.toml` | 构建元数据、包依赖、可选 extras、CLI 脚本、uv source/index 配置、setuptools package data、Ruff、Bandit、typos 和 mypy 设置。 |
| `uv.lock` | 锁定依赖版本。 |
| `requirements.in` | 兼容用 requirement 输入文件。 |
| `docs-requirements.txt` | 文档构建依赖。 |
| `requirements-ubuntu.txt`, `requirements-macos.txt` | 操作系统级安装参考。 |
| `Makefile` | Docker 镜像目标，以及 ACT、Diffusion、TD-MPC、SmolVLA 的端到端 train/eval 测试 recipe。 |
| `MANIFEST.in` | source distribution 包含规则。 |
| `setup.py` | setuptools 兼容 setup 入口。 |
| `LICENSE`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `AI_POLICY.md` | 项目许可、安全、贡献和治理文档。 |
| `README.md` | 项目概览和用户 quickstart。 |
| `AGENTS.md` | 本仓库中 AI agent 的开发指导。 |
| `AGENT_GUIDE.md` | 帮助用户使用 LeRobot 时面向 agent 的实用命令指南。 |

运行时 CLI 由 `pyproject.toml` 的 `[project.scripts]` 安装，全部指向 `src/lerobot/scripts/*.py`。

## 测试和质量检查

从仓库文件中确认到的主要命令：

```bash
uv run pytest tests -svv --maxfail=10
DEVICE=cuda make test-end-to-end
pre-commit run --all-files
```

质量/工具配置：

- Ruff 目标 Python 3.12，行宽 110，使用双引号，isort first-party package 为 `lerobot`。
- Bandit 已配置，但跳过 tests、benchmarks 和部分规则 ID。
- Mypy 是渐进式的：大部分 `lerobot.*` 忽略错误，但 `envs`、`configs`、`optim`、`model`、`cameras`、`motors`、`transport` 检查更严格。
- `tests/artifacts/**/*.safetensors` 和 protobuf 生成文件被排除在 Ruff 之外。

`Makefile` 中的 E2E 测试会在 `tests/outputs/` 下为代表性策略和环境创建短训练/评估运行。

## 生成的、重复的或低信号文件

- `uv.lock` 由 uv 生成，不应手动编辑。
- `src/lerobot/transport/services_pb2.py` 和 `services_pb2_grpc.py` 由 `services.proto` 生成。
- `tests/artifacts/` 包含测试使用的二进制 fixture（`.safetensors`、`.mp4`、`.png`、`.bag`）；应按路径/用途理解，不要当作源码阅读。
- `media/readme/` 包含 README 视觉资产。
- 策略子目录重复 `configuration_*`、`modeling_*`、`processor_*` 模式；除非正在修改特定策略，否则按策略家族理解即可。
- 机器人/遥操作子目录按每个硬件家族重复 `config_*` 加实现文件的模式。
- `docs/source/` 中有大量主题页面；`_toctree.yml` 是导航锚点。
- `src/lerobot.egg-info/` 和 `__pycache__/` 是当前 checkout 中存在的构建/运行产物，不是事实来源。

## 常见修改路径

| 目标 | 从这里开始 | 同时检查 |
| --- | --- | --- |
| 修改训练行为 | `src/lerobot/scripts/lerobot_train.py`, `src/lerobot/configs/train.py` | `src/lerobot/common/train_utils.py`, `tests/training/`, `Makefile` |
| 新增或修改策略 | `src/lerobot/policies/<policy>/`, `src/lerobot/policies/factory.py` | `src/lerobot/configs/policies.py`, `tests/policies/`, 策略文档 |
| 修改数据集加载/写入 | `src/lerobot/datasets/lerobot_dataset.py` | `dataset_reader.py`, `dataset_writer.py`, `video_utils.py`, `tests/datasets/` |
| 新增数据集工具 | `src/lerobot/datasets/dataset_tools.py`, `src/lerobot/scripts/lerobot_edit_dataset.py` | `tests/datasets/test_dataset_tools.py`, `docs/source/using_dataset_tools.mdx` |
| 新增 processor | `src/lerobot/processor/pipeline.py`, 相关 processor 文件 | `tests/processor/`, `docs/source/implement_your_own_processor.mdx` |
| 新增环境 | `src/lerobot/envs/configs.py`, `src/lerobot/envs/factory.py` | `tests/envs/`, `docs/source/envhub.mdx`, `pyproject.toml` 中的可选依赖 |
| 新增机器人硬件 | `src/lerobot/robots/robot.py`, `src/lerobot/robots/<name>/` | `src/lerobot/motors/`, `src/lerobot/cameras/`, `tests/robots/`, 硬件文档 |
| 新增遥操作硬件 | `src/lerobot/teleoperators/teleoperator.py`, `src/lerobot/teleoperators/<name>/` | `tests/teleoperators/`, record/teleoperate 脚本 |
| 修改相机支持 | `src/lerobot/cameras/` | `tests/cameras/`, `pyproject.toml` 中的可选依赖 |
| 修改电机驱动 | `src/lerobot/motors/` | `tests/motors/`, setup/calibration 脚本 |
| 修改文档导航 | `docs/source/_toctree.yml` | 相关 `.mdx` 页面和策略 README 文档 |
| 修改 CLI 可用性 | `pyproject.toml` 中的 `[project.scripts]` | `src/lerobot/scripts/` 中对应文件 |
| 修改 CI/测试行为 | `.github/workflows/`, `Makefile` | `pyproject.toml` 工具配置 |

## 未确认点和注意事项

- 一些模拟器集成刻意没有作为普通 PyPI extras 暴露，因为它们的依赖图或分发方式会和基础项目冲突；具体安装方式见 `pyproject.toml` 注释和相关 docs 页面。
- 面向硬件的测试可能会因平台、连接设备或可选依赖缺失而 skip。
- 二进制 fixture 和生成的 protobuf 文件是根据文件名、路径和引用关系分类的，没有做深度二进制解析。
- 本指南来自仓库静态检查，没有执行完整测试套件。
