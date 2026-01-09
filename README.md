# Raptor Foundation Policy 复现指南 (Docker 版)

本文档记录了从零开始复现 Raptor 基础策略（Foundation Policy）训练的全过程。本指南基于官方文档，但**修正了原仓库中的路径错误、脚本依赖缺失以及 CMake 构建配置问题**，确保在标准的 Docker 环境中可以顺利跑通全流程。

## 📋 目录
1. [主机环境准备](#1-主机环境准备)
2. [构建系统修复与编译](#2-构建系统修复与编译-关键)
3. [阶段一：预训练 (Pre-Training)](#3-阶段一预训练-pre-training)
4. [阶段二：蒸馏 (Post-Training)](#4-阶段二蒸馏-post-training)
5. [阶段三：验证与测试](#5-阶段三验证与测试)

---

## 1. 主机环境准备

在您的服务器终端（Host）执行以下操作，下载代码并准备 Docker 环境。

### 1.1 下载代码与数据

# 1. 克隆仓库
git clone [https://github.com/rl-tools/raptor.git](https://github.com/rl-tools/raptor.git)
cd raptor
git submodule update --init rl-tools
cd rl-tools
git submodule update --init --recursive -- external/highfive external/json external/tensorboard
cd ..

# 2. 准备训练数据 (合并分卷压缩包)
cat data/foundation-policy-v1-data.tar.gz.part_* > data/foundation-policy-v1-data.tar.gz
cd rl-tools
tar -xvf ../data/foundation-policy-v1-data.tar.gz
cd ..



### 1.2 启动持久化 Docker 容器

**注意**：这里使用了 `--name` 固定容器名称，并去掉了 `--rm`，防止退出容器后数据和编译文件丢失。

```bash
docker run -it --name raptor_training -v $(pwd)/rl-tools:/rl-tools -v data:/data rltools/rltools:ubuntu24.04_mkl_gcc_base

```

---

## 2. 构建系统修复与编译 [关键]

以下所有命令均在 **Docker 容器内部** (`root@...:/#`) 执行。

由于官方 CMake 配置在纯 Docker 环境下存在依赖定义的顺序问题，我们需要先手动修补构建文件。

### 2.1 修补 CMakeLists.txt

我们需要在 `src/CMakeLists.txt` 中手动定义核心库目标，防止编译子模块时报错 `Target links to RLtools::RLtools but the target was not found`。

```bash
# 使用 sed 在文件顶部插入必要的库定义
sed -i '1i\add_library(RLtools::Core INTERFACE)\nadd_library(RLtools::RLtools INTERFACE)' src/CMakeLists.txt

```

### 2.2 配置与编译

```bash
# 1. 创建并进入构建目录
mkdir -p build && cd build

# 2. 清理环境 (防止旧缓存干扰)
rm -rf *

# 3. 配置 CMake (指向 /rl-tools 根目录)
MKL_ROOT=/opt/intel/oneapi/mkl/latest cmake -S /rl-tools -B /build \
    -DCMAKE_BUILD_TYPE=Release \
    -DRL_TOOLS_BACKEND_ENABLE_MKL=ON \
    -DRL_TOOLS_ENABLE_TARGETS=ON \
    -DRL_TOOLS_EXPERIMENTAL=ON \
    -DRL_TOOLS_ENABLE_HDF5=ON \
    -DRL_TOOLS_ENABLE_JSON=ON \
    -DRL_TOOLS_ENABLE_TENSORBOARD=ON

# 4. 编译预训练程序 (使用多核并行编译)
cmake --build . --target foundation_policy_pre_training_sample_dynamics_parameters --target foundation_policy_pre_training -j$(nproc)

```

---

## 3. 阶段一：预训练 (Pre-Training)

### 3.1 生成动力学参数

生成 1000 组不同的无人机物理参数（如质量、电机响应等），用于领域随机化。

```bash
cd /rl-tools
export RL_TOOLS_EXTRACK_EXPERIMENT=$(date '+%Y-%m-%d_%H-%M-%S')
echo "当前实验ID: $RL_TOOLS_EXTRACK_EXPERIMENT"

/build/src/foundation_policy/foundation_policy_pre_training_sample_dynamics_parameters

```

### 3.2 并行运行预训练 (高速模式)

官方文档的串行命令非常慢。我们使用 `xargs -P` 并行训练 1000 个“教师”模型。这比串行快约 16 倍（取决于 CPU 核心数）。

```bash
# 注意：如果您 CPU 核心少于 16 个，请将 -P 16 改为 -P 8 或更小以避免卡顿
find src/foundation_policy/dynamics_parameters | sort | grep 'json$' | \
MKL_NUM_THREADS=1 RL_TOOLS_EXTRACK_EXPERIMENT=$RL_TOOLS_EXTRACK_EXPERIMENT \
xargs -I{} -P 16 /build/src/foundation_policy/foundation_policy_pre_training {}

```

*(此步骤耗时较长，请耐心等待直到所有任务完成)*

---

## 4. 阶段二：蒸馏 (Post-Training)

官方提供的 `extract_checkpoints.sh` 脚本在 Docker 中缺少依赖 (`p2s` 工具)，必须**手动执行**以下步骤。

### 4.1 手动生成教师模型列表

我们需要找到刚刚训练好的 1000 个模型，并生成一个格式为 `ID,STEP` 的列表文件。

```bash
cd /rl-tools

# 1. 查找所有 checkpoint -> 2. 提取ID和步数 -> 3. 排序 -> 4. 去重(只保留训练步数最大的)
find experiments -name "checkpoint.h5" \
| awk -F'/' '{print $(NF-4) "," $(NF-1)}' \
| sort -t, -k1,1n -k2,2rn \
| awk -F, '!seen[$1]++' \
> src/foundation_policy/checkpoints_$RL_TOOLS_EXTRACK_EXPERIMENT.txt

# 验证行数 (必须精确等于 1000)
wc -l src/foundation_policy/checkpoints_$RL_TOOLS_EXTRACK_EXPERIMENT.txt

```

### 4.2 修改 C++ 源代码

修改 `post_training/main.cpp` 以读取我们生成的新列表，并修正硬编码的路径错误。

```bash
# 自动修改 1: 替换 checkpoint 文件名为我们刚刚生成的列表文件名
sed -i "s|checkpoints_2025-04-16_20-10-58.txt|checkpoints_$RL_TOOLS_EXTRACK_EXPERIMENT.txt|g" src/foundation_policy/post_training/main.cpp

# 自动修改 2: 修复参数加载路径 (去掉时间戳后缀，指向固定目录)
sed -i 's|dynamics_parameters_" + checkpoint_path.experiment + "/"|dynamics_parameters/"|g' src/foundation_policy/post_training/main.cpp

# 自动修改 3: 替换实验目录 (从 "1k-experiments" 改为 "experiments")
sed -i 's|"1k-experiments"|"experiments"|g' src/foundation_policy/post_training/main.cpp

```

### 4.3 编译并运行蒸馏

```bash
# 1. 重新编译 Post-Training
cd /build
cmake --build . --target foundation_policy_post_training -j$(nproc)

# 2. 运行蒸馏程序
cd /rl-tools
/build/src/foundation_policy/foundation_policy_post_training

```

*(程序运行结束后，最终模型将保存在 `logs/` 目录下)*

---

## 5. 阶段三：验证与测试

为了验证训练结果，我们需要启用并修复默认被禁用的测试工具。

### 5.1 启用并修改测试工具代码

默认情况下测试工具未编译，且不支持命令行参数。

**1. 修改 CMakeLists.txt (取消注释)**

```bash
# 使用 sed 取消注释 foundation_policy_test_checkpoint 相关的两行
sed -i 's/^#\s*add_executable(foundation_policy_test_checkpoint/add_executable(foundation_policy_test_checkpoint/' src/foundation_policy/CMakeLists.txt
sed -i 's/^#\s*target_link_libraries(foundation_policy_test_checkpoint/target_link_libraries(foundation_policy_test_checkpoint/' src/foundation_policy/CMakeLists.txt

```

**2. 修改 C++ 代码 (支持传参)**
编辑 `src/foundation_policy/post_training/test_checkpoint.cpp`，使其能接收命令行参数。

```bash
# 将 int main() 改为 int main(int argc, char** argv)
sed -i 's/int main(){/int main(int argc, char** argv){/' src/foundation_policy/post_training/test_checkpoint.cpp

# 修改文件加载逻辑，优先使用 argv[1]
# 注意：这里使用 perl 进行多行/复杂替换，比 sed 更稳健
perl -i -pe 's|auto file = HighFive::File\(".*?", HighFive::File::ReadOnly\);|std::string model_path = (argc > 1) ? argv[1] : "logs/default_checkpoint.h5";\n    auto file = HighFive::File(model_path, HighFive::File::ReadOnly);|' src/foundation_policy/post_training/test_checkpoint.cpp

```

### 5.2 编译测试工具

```bash
cd /build
# 重新配置以识别新启用的目标
MKL_ROOT=/opt/intel/oneapi/mkl/latest cmake -S /rl-tools -B /build -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_BACKEND_ENABLE_MKL=ON -DRL_TOOLS_ENABLE_TARGETS=ON -DRL_TOOLS_EXPERIMENTAL=ON -DRL_TOOLS_ENABLE_HDF5=ON -DRL_TOOLS_ENABLE_JSON=ON -DRL_TOOLS_ENABLE_TENSORBOARD=ON

# 编译
cmake --build . --target foundation_policy_test_checkpoint -j$(nproc)

```

### 5.3 运行最终测试

```bash
cd /rl-tools

# 自动找到最新的蒸馏模型
FINAL_MODEL=$(ls -td logs/*/checkpoint.h5 | head -1)
echo "正在测试最终模型: $FINAL_MODEL"

# 运行测试
/build/foundation_policy/foundation_policy_test_checkpoint $FINAL_MODEL

```

### 预期结果

如果输出显示 `Mean return` 为较高的正数（例如 >100），`Mean episode length` 接近最大步数（例如 500），且 `Share terminated` 很低（<5%），则说明复现成功！

---


```

```
