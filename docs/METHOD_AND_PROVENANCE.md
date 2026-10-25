# 方法、版本、实验谱系与来源

## 固定版本

| 内容 | 版本 |
|---|---|
| 第三方完整参考实现 | `zita-ch/supermimic_vibed` @ `a1508d42c9d1699a52e8233ab16d08c488441ab9` |
| IsaacLab | `46d6cc3d7b115906fad12efd372fbf69f4e3d3ed` |
| Newton | `71eeba16a717d4dcf0b9dd7056a7b6f863054dc3` |
| Python | 3.12.3 |
| PyTorch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| Warp | 1.17.0官方cu12轮子 |
| MuJoCo / MuJoCo-Warp | 3.15.0 / 3.15.0 |
| 驱动与硬件 | 570.144 / RTX5090 32GB |
| 完整环境证据 | `environment/observed-freeze.txt` |

便携脚本由真实实验脚本整理，运行路径和GPU选择改为配置项。正式训练时代码快照内部提交为`a4e8e645f5f045fd1ca08d867756912f303073d2`，不是上游提交；训练改动可通过`patches/training_adapter.diff`审阅。上游原始`train.py` SHA256为`6f364bb904fb173a6d76f6da2af4bbe0eeb7dd51dd8370d35c7f067b117218c6`，当时生成训练适配文件SHA256为`05ecfea1a8300e6c1b0cd02ecff3af23e549bca0f36446a450d15cad796bf165`。

## V14采用的完整实现

保留上游策略/世界模型/critic网络、观测历史及归一化、未来目标编码、相对骨盆朝向输入、骨盆奖励锚点、next-frame奖励时序、解析奖励/终止、H32想象优化、k8世界模型拟合、GAE、噪声、DR和reset采样。默认4096环境、2048世界模型batch、4096策略batch，500次世界模型预热。

本仓库没有把V13镜像增强、强制动作对称、旧奖励补丁带入V14。V14所有网络重新随机初始化seed137，V12/V13只用于历史视频对比。

与参考仓库的必要适配：

1. 后空翻输入是50Hz，而上游默认读取30Hz LAFAN1；使用已有MotionLib构造参数`src_fps=50`，不重复插值。
2. 增加500次预热之后90分钟策略墙钟预算及最终检查点保存；driver总上限3小时。
3. 使用独立原生venv和CUDA12.8轮子，替代上游Docker CUDA13。主机驱动不升级。
4. 额外只读接触记录、严格工程评测及离线显示渲染，不改变训练奖励。

上游自身也与论文存在选择差异：自碰撞关闭、bf16 MLP计算、解析奖励/终止头、额外目标相对朝向等，见保留的`third_party/supermimic_vibed/README.md`。上游是非官方实现且作者明确提示未充分审查，不能称为官方代码。

## 历史排查与归因限制

早期V12/V13存在严重分腿和侧扭。已检验或否决的假设包括骨盆锚点单独替换、仅想象奖励时间修正、世界模型镜像数据增强、强制输出对称等：单个指标改善不足以采用，部分试验加重不对称或落地问题。

V14改为完整参考代码后，起跳两脚同步、姿态跟踪和侧扭明显改善。这说明整套参考实现提供了更可靠的单动作基线，**不能据此断言旧实现唯一错误是某个奖励项或自碰撞**。跨版本算法、机器人资产、碰撞配置、初始化与训练时长同时不同，不是严格单变量消融。

## 动作、机器人和许可

- 方法论文：[SuperMimic](https://arxiv.org/abs/2610.28480)。本仓库没有复制论文正文。
- 参考代码：[zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)，MIT，Copyright 2026 ETH Zurich, Chong Zhang；原版权保存在`third_party/.../LICENSE`和`licenses/SUPERMIMIC_MIT_LICENSE`。
- 后空翻目标：[kevinzakka/mjbatch](https://github.com/kevinzakka/mjbatch/blob/77966f85bcd8f7ef4351cb4a1a6f42e133d19725/examples/assets/flip.npz)，来源仓库Apache-2.0（许可证保存在`licenses/MJBATCH_LICENSE`）。源文件Git blob `da773fc61430e7951ca8e56832816051bdd44a72`，SHA256 `f9ad1275f978bf891ee141e272d8279eb56685e4209f42d5cd7dc23529bcb4d6`。下载后按复现文档裁剪与静态补帧，明确记录了修改；不重新发布原始NPZ。
- G1 URDF/网格：[lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main/robot_description/g1)，上游说明机器人描述来自Unitree（BSD-3-Clause）。此处只提供下载URL与哈希，机器人资产不打包，适用原来源许可。
- 本实验没有使用该数据集的40条LAFAN1动作；不要把其CC BY-NC-ND动作许可与这里的mjbatch后空翻目标混淆。
- IsaacLab、Newton、MuJoCo-Warp使用各自许可证；安装时从对应仓库获取。本项目MIT许可证不覆盖第三方资产，也不改变它们的使用条件。
- 本次训练输出权重、实际仿真视频和指标属于本次实验成果；未使用外部训练好的策略作为初始化。Release权重供复评，不应拿来宣称新的从零训练成果。

## 可复现性验证范围

原Linux环境完成了正式训练、最终评测、200Hz受力提取、对比渲染与蓝色棋盘格渲染。仓库整理后的便携代码另做语法检查、训练adapter哈希比对、数据重建哈希检查和现有Linux运行环境中的独立策略评测。具体验证结果见`results/packaging_validation.json`。未重新进行第二次90分钟训练，也未在全新机器从空目录重装全部依赖；不承诺跨驱动逐位一致或同时间得到同迭代数。
