# 从零复现步骤

## 1. 硬件和系统

已验证实验平台：Linux x86_64、RTX5090 32GB、NVIDIA驱动570.144、Python3.12.3。训练仅使用一张显卡。建议准备至少30GB空闲磁盘用于依赖、编译缓存、检查点和视频；具体占用随缓存变化。

系统需要Git、ffmpeg、gcc/g++、EGL/OpenGL运行库、DejaVu字体、Python3及uv。Ubuntu示例（系统安装由使用者执行）：

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg build-essential libegl1 libgl1 fonts-dejavu-core python3
# 按uv官方说明安装：https://docs.astral.sh/uv/getting-started/installation/
```

本项目不会更新显卡驱动或修改已有Conda环境。不要给Warp施加`ulimit -v`这类虚拟地址空间上限；其CUDA地址预留可能远大于实际物理内存。

## 2. 固定源码并安装环境

克隆本仓库后进入根目录。默认运行文件写入`.runtime/`；可设置绝对路径`BACKFLIP_WORKDIR`，之后每一步都要保持一致。

```bash
python3 scripts/setup.py
```

安装脚本会克隆固定版本IsaacLab和Newton、创建独立Python3.12虚拟环境、安装Torch2.11.0/CUDA12.8及Warp1.17.0官方cu12轮子，并按实际冻结记录安装其余包。不需要Docker或Isaac Sim GUI。轮子带SHA256验证。

原实验环境安装及训练已验证；**新整理的安装脚本没有在一台全新空白主机上完整重装验证**。固定依赖中包含开发版/预发布包，若镜像缺包请使用脚本指定的官方索引，不要无记录地升级版本。`environment/observed-freeze.txt`是版本证据，不可直接`pip install -r`：其中可移植占位路径由setup脚本转换。

## 3. 明确选择一张GPU

```bash
nvidia-smi --query-gpu=index,uuid,name --format=csv
export CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
export MUJOCO_EGL_DEVICE_ID=1  # 改成上面UUID对应的物理序号；单卡机器通常为0
export OMP_NUM_THREADS=1
```

不要照抄某台服务器的UUID。`CUDA_VISIBLE_DEVICES`控制训练设备，进程内部使用`cuda:0`；EGL物理序号用于视频渲染。脚本要求显式选择且只暴露一张GPU。

## 4. 准备机器人与动作

```bash
.runtime/.venv/bin/python scripts/prepare_data.py
python3 scripts/build_baseline_adapter.py
```

`prepare_data.py`从上游指定的Hugging Face数据集下载G1 URDF和38个网格，逐文件校验SHA256，再调用原仓库转换工具生成MJCF和USD。机器人资产不打包在本仓库。

后空翻动作来自`kevinzakka/mjbatch/examples/assets/flip.npz`，固定commit和SHA256。取原始帧30–240（含端点，共211帧），前加10个首帧、后加100个末帧，共321帧，50Hz。输出CSV为根位置3、根四元数xyzw4、29关节角。**它是动作目标，不是策略权重**。

生成CSV哈希必须与本次训练输入完全一致。原轨迹321个采样点覆盖6.4秒；评测执行321个控制步后再保持5秒，总轨迹约11.42秒。静态首尾帧用于静止起步与目标保持，不能把它理解为已训练动作。

`build_baseline_adapter.py`保留上游`train.py`，生成`train_bounded_backflip.py`。适配仅有50Hz动作输入、90分钟策略训练计时及结束时保存最后检查点；diff和SHA256写入运行目录。

## 5. 接触力校验与容量测试

```bash
.runtime/.venv/bin/python scripts/probe_reference_env.py
.runtime/.venv/bin/python scripts/run_reference_training.py smoke
```

先检查`.runtime/validation/reference_env_probe.json`的`passed:true`，再检查`.runtime/smoke_result.json`的`exit_code:0`和`finite_metrics:true`。

静态校验使用**仅用于probe的水平/姿态夹具和关节静力前馈，竖直外部支撑为零**。它验证受力读取是否正确，不代表未训练控制器能站稳。此夹具绝不会进入训练或策略评测。详见评测文档。

容量smoke使用完整4096环境/原batch，但只有2次更新，没有完成预热，不能展示为成功后空翻。首次torch编译可能耗时数分钟。

## 6. 正式随机初始化训练

```bash
.runtime/.venv/bin/python scripts/run_reference_training.py main
```

固定seed137、500次世界模型预热；之后策略预算90分钟，driver总上限3小时。没有resume参数，所有网络从随机初始化开始。正式run默认：

```
.runtime/runs/backflip_v14_full_reference_seed137/
  train_log.csv
  ckpt_*.pt
  last.pt
  final_evaluation/
```

查看`.runtime/train_v14.log`和`.runtime/main_launch.json`。不要只依赖旧PID，也不要只看训练return判断翻转质量。

已有run会拒绝覆盖。需要独立重复实验时使用新的`BACKFLIP_WORKDIR`并重新完成数据与校验准备。停止/失败时保留目录诊断，不要删除检查点后伪装为同一次运行。

## 7. 最后检查点评测与视频

正常完成后driver自动执行以下流程；也可在训练结束后手动重跑到新目录：

```bash
.runtime/.venv/bin/python scripts/evaluate_reference_backflip.py \
  --checkpoint .runtime/runs/backflip_v14_full_reference_seed137/last.pt \
  --output .runtime/independent_evaluation
.runtime/.venv/bin/python scripts/analyze_takeoff_forces.py .runtime/independent_evaluation
.runtime/.venv/bin/python scripts/render_blue_floor.py .runtime/independent_evaluation
```

输出包含`metrics.json`、`asymmetry_metrics.json`、`rollout.npz`、`contact_forces_200hz.npz`、`takeoff_force_metrics.json`、受力图及`blue_floor/policy_blue_checker.mp4`。视频每隔一个50Hz真实状态渲染一帧，25fps，保持原时间轴。仅调用FK生成画面，没有再次求解或修改原轨迹。

查看Release权重：下载`v14_final9029_last.pt`并核验`SHA256SUMS`，将上面的`--checkpoint`改为该文件。它是**本次训练的输出**，不是从零训练的输入。PyTorch检查点含优化器/配置，评测用`weights_only=False`；只加载可信来源文件。

正式验收通过必须再独立评测并检查起跳、腾空、落地与保持段视频。本次记录仍有1帧几何穿透，因此未宣称通过，无论原生return多高。

## 常见问题

- `geom1/geom2`显示全部为0：MuJoCo-Warp回传填充的是`contact.geom`数组；本记录器已使用该字段。
- `libEGL`或无显示错误：确认系统EGL/NVIDIA配置及`MUJOCO_EGL_DEVICE_ID`，不要为渲染改动策略。
- USD导入失败：在独立进程运行上游`tools/mjcf_to_usd.py`，该脚本注册所需schema；不要在已运行仿真里转换。
- 显存不足：先检查是否有其他任务，不能擅自结束他人进程。改变环境数/batch是新实验，不再等同于本基线。
- 最终迭代不是9029：90分钟墙钟预算和GPU非确定性所致；保留真实最后权重，不能挑选旧权重冒充最后结果。
