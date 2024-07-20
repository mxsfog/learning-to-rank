# Ranking Models and Utilities

This repository contains implementations of various models and utility functions for ranking tasks. The primary model is the Kernel-based Neural Ranking Model (KNRM). Other files include utility functions for gradient boosting, loss calculation, and metric evaluation.

## File Descriptions

### `knrm.py`

This file contains the implementation of the Kernel-based Neural Ranking Model (KNRM). KNRM leverages kernels to capture the soft matches between query and document terms and uses a multi-layer perceptron (MLP) for ranking.

#### Key Components

- **GaussianKernel**: A class defining a Gaussian kernel used to compute soft matches between query and document terms.
- **KNRM**: The main class implementing the KNRM model, which includes embedding layers, kernel layers, and an MLP for scoring.
- **RankingDataset**: A base class for datasets used in ranking tasks, providing common functionality for tokenizing and indexing texts.
- **TrainTripletsDataset** and **ValPairsDataset**: Classes extending `RankingDataset` for handling triplet and pair data formats respectively.
- **collate_fn**: A function for collating batches of data during training and validation.
- **Solution**: A class that encapsulates the entire training and evaluation process for the KNRM model, including data loading, preprocessing, and model evaluation.

### `gradient_boost.py`

This file contains implementations related to gradient boosting algorithms. It includes custom boosting techniques tailored for ranking tasks.

### `loss_funcs.py`

This file provides various loss functions used in ranking tasks. These loss functions are crucial for training models by defining how the model's predictions are penalized when they deviate from the true rankings.

### `metric_funcs.py`

This file includes various metric functions to evaluate the performance of ranking models. Key metrics include NDCG (Normalized Discounted Cumulative Gain) and other ranking-specific measures.

## Installation

To use the code in this repository, you'll need to install the required Python packages. You can install them using pip:

```bash
pip install -r requirements.txt
```

## Usage

### KNRM Model

1. **Prepare Data**: Ensure your data is in the correct format (e.g., GLUE QQP dataset for training).
2. **Initialize Solution**: Create an instance of the `Solution` class with appropriate parameters.
3. **Train Model**: Call the `train` method on the `Solution` instance to start training the model.

```python
from knrm import Solution

glue_qqp_dir = '/path/to/QQP/'
glove_path = 'path/to/glove.6B.50d.txt'

solution = Solution(glue_qqp_dir=glue_qqp_dir, glove_vectors_path=glove_path)
solution.train(n_epochs=20)
```

### Gradient Boosting

Refer to the `gradient_boost.py` file for custom gradient boosting implementations and usage examples.

### Loss Functions

Import and use the loss functions from `loss_funcs.py` in your training scripts as needed.

### Metric Functions

Use the metrics from `metric_funcs.py` to evaluate your ranking models. For example:

```python
from metric_funcs import ndcg_k

# Example usage
ys_true = [3, 2, 3, 0, 1, 2]
ys_pred = [0.6, 0.4, 0.7, 0.3, 0.2, 0.1]
ndcg_score = ndcg_k(ys_true, ys_pred, k=5)
print(f'NDCG@5: {ndcg_score}')
```

