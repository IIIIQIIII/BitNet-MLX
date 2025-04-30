# BitNet-MLX

BitNet-MLX is a repository for running BitNet models using the MLX framework, optimized for Apple Silicon and compatible with other platforms. This guide shows how to set up and run BitNet-MLX on Google Colab.

## Prerequisites
- Google Colab environment.

## Setup and Running on Colab

Follow these steps to run BitNet-MLX on Google Colab:

1. **Clone the Repository**:
   ```bash
   !git clone https://github.com/IIIIQIIII/BitNet-MLX.git
   %cd /content/BitNet-MLX
   ```

2. **Install Dependencies**:
   ```bash
   !pip install -r requirements.txt
   ```

3. **Convert Model Weights**:
   Convert the default BitNet model weights to MLX format:
   ```bash
   !python convert.py
   ```

4. **Run Inference**:
   Generate text using the converted model with a sample prompt:
   ```bash
   !python inference.py --prompt "Once upon a time, in a land far away," --max-tokens 50 --temp 0.8
   ```

## Notes
- The default model is `1bitLLM/bitnet_b1_58-xl` with `float32` weights. Modify `convert.py` parameters for other models or data types.
- Ensure sufficient disk space for model weights (~several GB).
- Inference performance depends on Colab's hardware (CPU/GPU).

## Acknowledgments
Special thanks to the [mlx-bitnet](https://github.com/exo-explore/mlx-bitnet/tree/main) project.