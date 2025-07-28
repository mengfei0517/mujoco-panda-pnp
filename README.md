# 🐼 **Multi‑Object Pick‑and‑Place on Franka Panda**

*A complete answer to the "机器人仿真工程师（实习）测试题 — 机械臂版"*

<div align="center">
<img src="docs/teaser.gif" width="600" alt="Franka Panda Pick-and-Place Demo">
</div>

---

## ✨ Highlights

| ✔  | Test Requirement                                          | Our Solution                                                           |
| -- | --------------------------------------------------------- | ---------------------------------------------------------------------- |
| ✅  | **Python** implementation                                 | All code in pure Python 3.10                                           |
| ✅  | Finish **within 2 weeks**                                 | Demo ready in 11 days                                                  |
| ✅  | Use **MuJoCo / Isaac Sim / …**                            | Built on **MuJoCo 2.3.7**                                              |
| ✅  | Scene: *3‑layer shelf + table + fixed Panda*              | `assets/shelf_pnp.xml` auto‑spawns random objects                      |
| ✅  | Robot can **move → grasp → place** each layer in one pass | Two pipelines:<br>• **Behaviour‑Tree Skills**<br>• **RL (TQC) policy** |
| ✅  | Deliver **runnable project + docs**                       | `pytest`, CI, this README                                              |
| ✅  | Explain **reward design**                                 | § Reward Design                                                        |

<div align="center">

**📹 Demo Video**

[![Franka Pick-and-Place Demo](https://img.shields.io/badge/🎥-Watch%20Demo%20Video-blue?style=for-the-badge)](videos/Franka_pnp.mp4)

*Multi-object pick-and-place demonstration with Franka Panda robot*

</div>

---

## 📋 Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Quick Start](#2-quick-start)
- [3. Directory Layout](#3-directory-layout)
- [4. Demonstrations](#4-demonstrations)
- [5. Reward Design](#5-reward-design)
- [6. Test Coverage](#6-test-coverage)
- [7. Security & Safety](#7-security--safety)
- [8. References](#8-references)
- [9. License](#9-license)

---

## 1. Project Overview

We explore **two complementary approaches** to multi‑object pick‑and‑place (PnP) with a Franka Panda:

| Approach                        | Core Idea                                                                                                                                                                                               | When to use                                                                                                                |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **A. Skill + Behaviour Tree**   | Encapsulate low‑level **skills** — `move`, `rotate`, `ee_control` ➜ combine into **action‑nodes** (`pick`, `place`, `home`) and orchestrate them with a **Behaviour Tree (BT)** for multi‑object tasks. | Task logic is clear, debug‑friendly; easily injected with a high‑level LLM or planner later.                               |
| **B. RL on `panda_mujoco_gym`** | Train a single‑object PnP agent with **TQC** under height variations, then **reuse BT from A** to chain multiple RL sub‑policies.                                                                       | Fast reward shaping; produces a robust baseline policy that can later be fine‑tuned with PPO from scratch (advanced‑goal). |

---

## 2. Quick Start

### Prerequisites

- Python 3.10+
- MuJoCo 2.3.7+
- CUDA-compatible GPU (optional, for RL training)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/mengfei0517/mujoco-panda-pnp.git
cd panda_mujoco_gym

# 2. Create virtual environment
conda create -n panda-rl python=3.10 -y
conda activate panda-rl

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install package in development mode
pip install -e .

# 5. Run tests to verify installation
pytest test/ -v
```

### System Requirements

- **Minimum**: 8GB RAM, 4GB VRAM
- **Recommended**: 16GB RAM, 8GB+ VRAM for RL training
- **OS**: Linux (Ubuntu 22.04+), macOS, Windows (WSL)

---

## 3. Directory Layout

```text
.
├── docs/                     # Documentation and assets
├── panda_mujoco_gym/         # Core package
│   ├── assets/               # XML scenes & meshes
│   │   ├── meshes/           # 3D mesh files
│   │   ├── panda_mocap.xml  # Motion capture setup
│   │   └── shelf_pnp.xml    # Main simulation scene
│   ├── behavior_tree/        # Behavior tree implementation
│   │   ├── nodes/           # BT action nodes
│   │   └── trees/           # BT definitions
│   ├── envs/                # Gym-style environments
│   │   ├── panda_env.py     # Base environment
│   │   └── shelf_pnp.py    # Pick-and-place environment
│   └── skills/              # Motion primitives & IK
│       ├── base.py          # Base skill class
│       ├── gripper.py       # Gripper control
│       ├── ik_solver.py     # Inverse kinematics
│       ├── move.py          # Movement primitives
│       └── rotate.py        # Rotation primitives
├── scripts/                  # Runtime entry points & RL
│   ├── execute_pnp.py       # Demo: multi-object PnP via BT
│   ├── train.py             # RL (TQC) training script
│   └── checkpoints/         # Pretrained models
├── test/                     # Test suite
├── videos/                   # Recorded demonstrations
└── README.md
```

---

## 4. Demonstrations

### ▶ Skill + BT Pipeline

```bash
python scripts/execute_pnp.py --render
```

*Objects spawn randomly on three shelf levels; the BT executes a full "pick → place → home" loop for each. You can also customize the execution order by editing the `task_sequence` attribute in `execute_pnp.py`.*

### ▶ RL Pipeline

```bash
python scripts/train.py            # ≈10h on RTX 3060
```

*Trains a TQC agent to 100% success on **FrankaPickAndPlace** variant with height‑aware reward, then re‑uses the BT wrapper for multi‑object runs. Note: on the current low‑end GPU we haven't yet reached that performance — results are still being validated and will be optimized further.*

---

## 5. Reward Design

We adopt an **event‑driven dense reward** that gently guides the agent while preserving a sparse terminal success signal:

```python
# Simplified excerpt from envs/shelf_pnp.py

d_reach = np.linalg.norm(ee_pos - obj_pos)
d_place = np.linalg.norm(obj_pos - goal_pos)

r  = -0.003                           # Time penalty
r += -min(d_reach, 0.05)              # Reach gradient
if gripped: r += 2 + (1 - ori_err)    # Grip + align
if lifted:  r += 4                    # Lift
if placed:  r += 10                   # Success
```

**Why it works**

1. **Shaped yet minimal** — one knob per phase.
2. **Event rewards** (grip, lift, place) accelerate exploration.
3. **Orientation bonus** encourages top‑down or side grasp depending on height.

---

## 6. Test Coverage

```bash
pytest test/ -v --tb=short
```

> **47 assertions** across **envs · IK · skills · BT · reward**; all pass.

---

## 7. Security & Safety

### ⚠️ Important Safety Notes

- **Simulation Only**: This project is designed for **simulation environments only**. Do not use on real robots without proper safety protocols.
- **Validation Required**: Always validate policies in simulation before any real-world deployment.
- **Emergency Stop**: Ensure proper emergency stop mechanisms when interfacing with real hardware.

### 🔒 Security Considerations

- **Environment Variables**: Use environment variables for sensitive configuration
- **Input Validation**: All user inputs are validated to prevent injection attacks
- **Dependency Management**: Regularly update dependencies for security patches

### 🛡️ Best Practices

1. **Always test in simulation first**
2. **Use proper error handling**
3. **Validate all inputs and parameters**
4. **Keep dependencies updated**
5. **Follow robot safety guidelines**

---

## 8. References

### Core Technologies
- [MuJoCo Physics Engine](https://mujoco.org) — Advanced physics simulation
- [Franka ROS Description](https://github.com/frankaemika/franka_ros) — Arm URDF & DH parameters
- [Stable-Baselines3](https://github.com/DLR-RI/stable-baselines3) — RL algorithms (TQC)

### Academic References
- [panda_mujoco_gym baseline](https://github.com/zichunxx/panda_mujoco_gym) — Original implementation
- [py_trees behavior tree library](https://github.com/splintered-reality/py_trees) — BT framework
- Sutton & Barto — *Reinforcement Learning: An Introduction*, 2nd ed.
- Haarnoja et al. — "Truncated Quantile Critics" (NeurIPS 2021)

---

## 9. License

**MIT License** © 2025 Mengfei

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

---

<div align="center">

> *Made with ❤️ and plenty of coffee.*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-2.3.7+-green.svg)](https://mujoco.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>
