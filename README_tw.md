# Open R1

*一個完全開源的 DeepSeek-R1 復現項目。此倉庫仍在開發中，歡迎共同參與！*

**目錄**  
1. [概覽](#概覽)  
2. [實施策略](#實施策略)  
3. [安裝指南](#安裝指南)  
4. [訓練模型](#訓練模型)  
   - [監督微調 (SFT)](#監督微調-sft)  
   - [GRPO 訓練](#grpo-訓練)  
5. [模型評估](#模型評估)  
6. [復現 DeepSeek 在 MATH-500 的評估結果](#復現-deepseek-在-math-500-的評估結果)  
7. [數據生成](#數據生成)  
   - [從小型蒸餾 R1 模型生成數據](#從小型蒸餾-r1-模型生成數據)  
   - [從 DeepSeek-R1 生成數據](#從-deepseek-r1-生成數據)  
8. [貢獻指南](#貢獻指南)

## 概覽

本倉庫的目標是構建 R1 流程的缺失組件，讓所有人都能復現並基於此項目開發。項目設計簡潔，主要包含以下內容：

- `src/open_r1`：包含訓練、評估模型及生成合成數據的腳本：
  - `grpo.py`：使用 GRPO 在指定數據集上訓練模型。
  - `sft.py`：在數據集上執行簡單的監督微調 (SFT)。
  - `evaluate.py`： 在 R1 基準測試中評估模型。
  - `generate.py`：使用[Distilabel](https://github.com/argilla-io/distilabel)從模型生成合成數據。
- `Makefile`：包含運行R1管道每個步驟的易於運行的命令，利用上述腳本。

### 實施策略

我們以 DeepSeek-R1 的[技術報告](https://github.com/deepseek-ai/DeepSeek-R1)為指南，大致分為三個主要步驟：

* **步驟 1**：通過從 DeepSeek-R1 蒸餾高質量語料，復現 R1-Distill 模型。
* **步驟 2**：復現 DeepSeek 用於創建 R1-Zero 的純強化學習流程。這可能涉及構建數學、推理和代碼的大規模數據集。
* **步驟 3**：展示通過多階段訓練從基礎模型到 RL 調優的過程。

<center>
    <img src="assets/plan-of-attack.png" width="500">
</center>

## 安裝指南

**注意：依賴庫需 CUDA 12.1 支持。若遇到段錯誤，請檢查系統配置。**

**創建 Python 虛擬環境**（推薦使用 `uv`）：
安裝 `uv`（參見 [UV 安裝指南](https://docs.astral.sh/uv/getting-started/installation/)）。

```shell
uv venv openr1 --python 3.11 && source openr1/bin/activate && uv pip install --upgrade pip
```

接下來安裝vLLM：

```shell
uv pip install vllm>=0.7.0

# 對於CUDA 12.1
pip install vllm>=0.7.0 --extra-index-url https://download.pytorch.org/whl/cu121
export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')"):$LD_LIBRARY_PATH
```

這也會安裝PyTorch `v2.5.1`，並且非常重要的是使用此版本，因為vLLM二進制文件是為其編譯的。然後可以通過`pip install -e .[LIST OF MODES]`安裝特定用例的其餘依賴項。對於大多數貢獻者，我們推薦：

```shell
pip install -e ".[dev]"
```

接下來，登錄到您的Hugging Face和Weights and Biases賬戶：

```shell
huggingface-cli login
wandb login
```

最後，檢查系統是否安裝了Git LFS，以便可以將模型/數據集加載並推送到Hugging Face Hub：

```shell
git-lfs --version
```

如果未安裝，請運行：

```shell
sudo apt-get install git-lfs
```

## 訓練模型

我們支持使用DDP或DeepSpeed（ZeRO-2和ZeRO-3）訓練模型。要切換方法，只需更改`configs`中`accelerate` YAML配置文件的路徑。

> [!NOTE]
> 下面的訓練命令配置為一個具有8 x H100（80GB）節點的硬體。對於不同的硬體和拓撲結構，可能需要調整批次大小和梯度累積步驟。

### 監督微調 (SFT)

要在從DeepSeek-R1精餾的數據集上運行SFT，該數據集包含推理跟蹤，例如[Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k)，請運行：

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py --config recipes/qwen/Qwen2.5-1.5B-Instruct/sft/config_full.yaml
```

要啓動Slurm作業，請運行：

```shell
sbatch --output=/path/to/logs/%x-%j.out --err=/path/to/logs/%x-%j.err slurm/sft.slurm {model} {dataset} {accelerator}
```

這裡`{model}`和`{dataset}`指的是Hugging Face Hub上的模型和數據集ID，而`{accelerator}`指的是`configs`中選擇的🤗 Accelerate配置文件。

### GRPO

要通過GRPO訓練器進行訓練，我們使用一個GPU運行vLLM以加快生成速度，其餘GPU用於訓練。例如，在一個有8個GPU的節點上，使用`recipes/accelerate_configs/zero3.yaml`配置文件，然後覆蓋`num_processes`以在7個設備上運行：

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml --num_processes=7 src/open_r1/grpo.py --config recipes/qwen/Qwen2.5-1.5B-Instruct/grpo/confg_full.yaml
```

要啓動Slurm作業，請運行：

```shell
sbatch --output=/path/to/logs/%x-%j.out --err=/path/to/logs/%x-%j.err slurm/grpo.slurm {model} {dataset} {accelerator}
```

您可以在[recipes](./recipes)中找到更多模型配置。

## 模型評估

我們使用`lighteval`來評估模型，並在`src/open_r1/evaluate.py`中定義自定義任務。對於可以在單個GPU上運行的模型，請運行：

```shell
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=float16,max_model_length=32768,gpu_memory_utilisation=0.8"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --system-prompt="請逐步推理，並將最終答案放置在\boxed{}中。" \
    --output-dir $OUTPUT_DIR
```

要在多個GPU上增加吞吐量，請使用_data parallel_：

```shell
NUM_GPUS=8
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=float16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.8"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --system-prompt="請逐步推理，並將最終答案放置在\boxed{}中。" \
    --output-dir $OUTPUT_DIR
```

對於需要在GPU上分片的大型模型，請使用_tensor parallel_並運行：

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
    --system-prompt="請逐步推理，並將最終答案放置在\boxed{}中。" \
    --output-dir $OUTPUT_DIR
```

您也可以通過`make evaluate`啓動評估，指定模型、任務以及可選的並行技術和GPU數量。

在單個GPU上評估：

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24
```

使用數據並行：

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24 PARALLEL=data NUM_GPUS=8
```

使用張量並行：

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24 PARALLEL=tensor NUM_GPUS=8
```

## 在MATH-500上復現DeepSeek的評估結果

我們能夠在MATH-500基準測試上復現DeepSeek報告的結果：

| 模型                      | MATH-500 (HF lighteval) | MATH-500 (DeepSeek報告) |
| :-------------------------- | :-------: | :----------------------------: |
| DeepSeek-R1-Distill-Qwen-1.5B  |  81.6   |              83.9              |
| DeepSeek-R1-Distill-Qwen-7B    |  91.8   |              92.8              |
| DeepSeek-R1-Distill-Qwen-14B   |  94.2   |              93.9              |
| DeepSeek-R1-Distill-Qwen-32B   |  95.0   |              94.3              |
| DeepSeek-R1-Distill-Llama-8B   |  85.8   |              89.1              |
| DeepSeek-R1-Distill-Llama-70B  |  93.4   |              94.5              |

要復現這些結果，請使用以下命令：

```shell
sbatch slurm/evaluate.slurm deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B math_500
sbatch slurm/evaluate.slurm deepseek-ai/DeepSeek-R1-Distill-Qwen-7B math_500
sbatch slurm/evaluate.slurm deepseek-ai/DeepSeek-R1-Distill-Qwen-14B math_500
sbatch slurm/evaluate.slurm deepseek-ai/DeepSeek-R1-Distill-Qwen-32B math_500 tp
sbatch slurm/evaluate.slurm deepseek-ai/DeepSeek-R1-Distill-Llama-8B math_500
sbatch slurm/evaluate.slurm deepseek-ai/DeepSeek-R1-Distill-Llama-70B math_500 tp
```

## 數據生成

### 從小型精餾R1模型生成數據

以下示例可以在1xH100上運行。首先安裝以下依賴項：

```shell
uv pip install "distilabel[vllm]>=1.5.2"
```

現在，將以下代碼段保存到名為`pipeline.py`的文件中，並使用`python pipeline.py`運行它。它將為每個10個示例生成4個輸出（請將用戶名庫的存儲庫更改為您組織/用戶名）：

```python
from datasets import load_dataset
from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration

prompt_template = """\
你將看到一個問題。請逐步推理，並將最終答案放在\boxed{}中：
{{ instruction }}"""

dataset = load_dataset("AI-MO/NuminaMath-TIR", split="train").select(range(10))

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # 用另一個小型精餾r1模型替換

with Pipeline(
    name="distill-qwen-7b-r1",
    description="從精餾r1模型生成數據的管道",
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

查看示例數據集在[HuggingFaceH4/numina-deepseek-r1-qwen-7b](https://huggingface.co/datasets/HuggingFaceH4/numina-deepseek-r1-qwen-7b)。

### 從DeepSeek-R1生成數據

要運行更大的DeepSeek-R1，我們使用了每個節點有8×H100 GPU的2個節點，使用存儲庫中位於`slurm/generate.slurm`的Slurm文件。首先安裝依賴項：

（目前我們需要安裝修復R1 CUDA圖形捕獲的vllm開發輪子）

```shell
pip install https://wheels.vllm.ai/221d388cc5a836fa189305785ed7e887cea8b510/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu121

uv pip install "distilabel[vllm,ray,openai]>=1.5.2"
```

然後運行以下命令：

```shell
sbatch slurm/generate.slurm \
    --hf-dataset AI-MO/NuminaMath-TIR \
    --temperature 0.6 \
    --prompt-column problem \
    --model deepseek-ai/DeepSeek-R1 \
    --hf-output-dataset username/r1-dataset
```

> [!NOTE]
> 在作業運行期間，您可以通過集群登錄節點在本地計算機上設置SSH隧道來訪問Ray儀錶盤，方法是運行`ssh -L 8265:ray_ip_head_node:8265 <login_node>`，然後瀏覽`http://localhost:8265`

## 貢獻

貢獻歡迎。請參閱https://github.com/huggingface/open-r1/issues/23。
```