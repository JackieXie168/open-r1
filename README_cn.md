# Open R1

*这是DeepSeek-R1的完全开源再现。此存储库是一个正在进行的项目，一起建设它吧！*

**目录**
1. [概述](#overview)
2. [计划](#plan-of-attack)
3. [安装](#installation)
4. [训练模型](#training-models)
   - [SFT](#sft)
   - [GRPO](#grpo)
5. [评估模型](#evaluating-models)
6. [在MATH-500上复现DeepSeek的评估结果](#reproducing-deepseeks-evaluation-results-on-math-500)
7. [数据生成](#data-generation)
   - [从小型精馏R1模型生成数据](#generate-data-from-a-smol-distilled-r1-model)
   - [从DeepSeek-R1生成数据](#generate-data-from-deepseek-r1)
8. [贡献](#contributing)

## 概述

此存储库的目标是构建R1管道中的缺失部分，以便每个人都可以复现并在其基础上进行构建。该项目设计简单，主要由以下部分组成：

- `src/open_r1`：包含训练和评估模型以及生成合成数据的脚本：
  - `grpo.py`：使用GRPO在给定数据集上训练模型。
  - `sft.py`：对模型进行简单的SFT操作。
  - `evaluate.py`：在R1基准测试上评估模型。
  - `generate.py`：使用[Distilabel](https://github.com/argilla-io/distilabel)从模型生成合成数据。
- `Makefile`：包含运行R1管道每个步骤的易于运行的命令，利用上述脚本。

### 计划

我们将使用DeepSeek-R1的[技术报告](https://github.com/deepseek-ai/DeepSeek-R1)作为指南，该报告大致可以分为三个主要步骤：

* 步骤1：通过精馏DeepSeek-R1的高质量语料库来复现R1-Distill模型。
* 步骤2：复现DeepSeek用于创建R1-Zero的纯RL管道。这可能需要为数学、推理和代码整理新的大型数据集。
* 步骤3：展示我们可以通过多阶段训练从基础模型到RL调整。

<center>
    <img src="assets/plan-of-attack.png" width="500">
</center>

## 安装

**注意：库依赖于CUDA 12.1。如果遇到段错误，请检查您的系统。**

要运行此项目的代码，首先使用例如`uv`创建一个Python虚拟环境。
要安装`uv`，请参阅[UV安装指南](https://docs.astral.sh/uv/getting-started/installation/)。

```shell
uv venv openr1 --python 3.11 && source openr1/bin/activate && uv pip install --upgrade pip
```

接下来安装vLLM：

```shell
uv pip install vllm>=0.7.0

# 对于CUDA 12.1
pip install vllm>=0.7.0 --extra-index-url https://download.pytorch.org/whl/cu121
export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')"):$LD_LIBRARY_PATH
```

这也会安装PyTorch `v2.5.1`，并且非常重要的是使用此版本，因为vLLM二进制文件是为其编译的。然后可以通过`pip install -e .[LIST OF MODES]`安装特定用例的其余依赖项。对于大多数贡献者，我们推荐：

```shell
pip install -e ".[dev]"
```

接下来，登录到您的Hugging Face和Weights and Biases账户：

```shell
huggingface-cli login
wandb login
```

最后，检查系统是否安装了Git LFS，以便可以将模型/数据集加载并推送到Hugging Face Hub：

```shell
git-lfs --version
```

如果未安装，请运行：

```shell
sudo apt-get install git-lfs
```

## 训练模型

我们支持使用DDP或DeepSpeed（ZeRO-2和ZeRO-3）训练模型。要切换方法，只需更改`configs`中`accelerate` YAML配置文件的路径。

> [!NOTE]
> 下面的训练命令配置为一个具有8 x H100（80GB）节点的硬件。对于不同的硬件和拓扑结构，可能需要调整批次大小和梯度累积步骤。

### SFT

要在从DeepSeek-R1精馏的数据集上运行SFT，该数据集包含推理跟踪，例如[Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k)，请运行：

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py --config recipes/qwen/Qwen2.5-1.5B-Instruct/sft/config_full.yaml
```

要启动Slurm作业，请运行：

```shell
sbatch --output=/path/to/logs/%x-%j.out --err=/path/to/logs/%x-%j.err slurm/sft.slurm {model} {dataset} {accelerator}
```

这里`{model}`和`{dataset}`指的是Hugging Face Hub上的模型和数据集ID，而`{accelerator}`指的是`configs`中选择的🤗 Accelerate配置文件。

### GRPO

要通过GRPO训练器进行训练，我们使用一个GPU运行vLLM以加快生成速度，其余GPU用于训练。例如，在一个有8个GPU的节点上，使用`recipes/accelerate_configs/zero3.yaml`配置文件，然后覆盖`num_processes`以在7个设备上运行：

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml --num_processes=7 src/open_r1/grpo.py --config recipes/qwen/Qwen2.5-1.5B-Instruct/grpo/confg_full.yaml
```

要启动Slurm作业，请运行：

```shell
sbatch --output=/path/to/logs/%x-%j.out --err=/path/to/logs/%x-%j.err slurm/grpo.slurm {model} {dataset} {accelerator}
```

您可以在[recipes](./recipes)中找到更多模型配置。

## 评估模型

我们使用`lighteval`来评估模型，并在`src/open_r1/evaluate.py`中定义自定义任务。对于可以在单个GPU上运行的模型，请运行：

```shell
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=float16,max_model_length=32768,gpu_memory_utilisation=0.8"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --system-prompt="请逐步推理，并将最终答案放置在\boxed{}中。" \
    --output-dir $OUTPUT_DIR
```

要在多个GPU上增加吞吐量，请使用_data parallel_：

```shell
NUM_GPUS=8
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=float16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.8"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --system-prompt="请逐步推理，并将最终答案放置在\boxed{}中。" \
    --output-dir $OUTPUT_DIR
```

对于需要在GPU上分片的大型模型，请使用_tensor parallel_并运行：

```shell
NUM_GPUS=8
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
MODEL_ARGS="pretrained=$MODEL,dtype=float16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.8"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

export VLLM_WORKER_MULTIPROC_METHOD=spawn
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --system-prompt="请逐步推理，并将最终答案放置在\boxed{}中。" \
    --output-dir $OUTPUT_DIR
```

您也可以通过`make evaluate`启动评估，指定模型、任务以及可选的并行技术和GPU数量。

在单个GPU上评估：

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24
```

使用数据并行：

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24 PARALLEL=data NUM_GPUS=8
```

使用张量并行：

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24 PARALLEL=tensor NUM_GPUS=8
```

## 在MATH-500上复现DeepSeek的评估结果

我们能够在MATH-500基准测试上复现DeepSeek报告的结果：

| 模型                      | MATH-500 (HF lighteval) | MATH-500 (DeepSeek报告) |
| :-------------------------- | :-------: | :----------------------------: |
| DeepSeek-R1-Distill-Qwen-1.5B  |  81.6   |              83.9              |
| DeepSeek-R1-Distill-Qwen-7B    |  91.8   |              92.8              |
| DeepSeek-R1-Distill-Qwen-14B   |  94.2   |              93.9              |
| DeepSeek-R1-Distill-Qwen-32B   |  95.0   |              94.3              |
| DeepSeek-R1-Distill-Llama-8B   |  85.8   |              89.1              |
| DeepSeek-R1-Distill-Llama-70B  |  93.4   |              94.5              |

要复现这些结果，请使用以下命令：

```shell
sbatch slurm/evaluate.slurm deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B math_500
sbatch slurm/evaluate.slurm deepseek-ai/DeepSeek-R1-Distill-Qwen-7B math_500
sbatch slurm/evaluate.slurm deepseek-ai/DeepSeek-R1-Distill-Qwen-14B math_500
sbatch slurm/evaluate.slurm deepseek-ai/DeepSeek-R1-Distill-Qwen-32B math_500 tp
sbatch slurm/evaluate.slurm deepseek-ai/DeepSeek-R1-Distill-Llama-8B math_500
sbatch slurm/evaluate.slurm deepseek-ai/DeepSeek-R1-Distill-Llama-70B math_500 tp
```

## 数据生成

### 从小型精馏R1模型生成数据

以下示例可以在1xH100上运行。首先安装以下依赖项：

```shell
uv pip install "distilabel[vllm]>=1.5.2"
```

现在，将以下代码段保存到名为`pipeline.py`的文件中，并使用`python pipeline.py`运行它。它将为每个10个示例生成4个输出（请将用户名库的存储库更改为您组织/用户名）：

```python
from datasets import load_dataset
from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration

prompt_template = """\
您将得到一个问题。请逐步推理，并将最终答案放置在\boxed{}中：
{{ instruction }}"""

dataset = load_dataset("AI-MO/NuminaMath-TIR", split="train").select(range(10))

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # 用另一个小型精馏r1模型替换

with Pipeline(
    name="distill-qwen-7b-r1",
    description="从精馏r1模型生成数据的管道",
) as pipeline:

    llm = vLLM(
        model=model_id,
        tokenizer=model_id,
        extra_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 8192,
        },
        generation_kwargs={
            "temperature": 0.6,
            "max_new_tokens": 8192,
        },
    )
    prompt_column = "problem"
    text_generation = TextGeneration(
        llm=llm,
        template=prompt_template,
        num_generations=4,
        input_mappings={"instruction": prompt_column} if prompt_column is not None else {}
    )

if __name__ == "__main__":
    distiset = pipeline.run(dataset=dataset)
    distiset.push_to_hub(repo_id="username/numina-deepseek-r1-qwen-7b")
```

查看示例数据集在[HuggingFaceH4/numina-deepseek-r1-qwen-7b](https://huggingface.co/datasets/HuggingFaceH4/numina-deepseek-r1-qwen-7b)。

### 从DeepSeek-R1生成数据

要运行更大的DeepSeek-R1，我们使用了每个节点有8×H100 GPU的2个节点，使用存储库中位于`slurm/generate.slurm`的Slurm文件。首先安装依赖项：

（目前我们需要安装修复R1 CUDA图形捕获的vllm开发轮子）

```shell
pip install https://wheels.vllm.ai/221d388cc5a836fa189305785ed7e887cea8b510/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu121

uv pip install "distilabel[vllm,ray,openai]>=1.5.2"
```

然后运行以下命令：

```shell
sbatch slurm/generate.slurm \
    --hf-dataset AI-MO/NuminaMath-TIR \
    --temperature 0.6 \
    --prompt-column problem \
    --model deepseek-ai/DeepSeek-R1 \
    --hf-output-dataset username/r1-dataset
```

> [!NOTE]
> 在作业运行期间，您可以通过集群登录节点在本地计算机上设置SSH隧道来访问Ray仪表盘，方法是运行`ssh -L 8265:ray_ip_head_node:8265 <login_node>`，然后浏览`http://localhost:8265`

## 贡献

贡献欢迎。请参阅https://github.com/huggingface/open-r1/issues/23。
```