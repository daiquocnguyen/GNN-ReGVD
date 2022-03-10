<p align="center">
	<img src="https://github.com/daiquocnguyen/GNN-ReGVD/blob/master/logo.png" width="125">
</p>

# Revisiting Graph Neural Networks for Vulnerability Detection

This program provides the implementation of our ReGVD, as described in [our paper](https://arxiv.org/abs/2110.07317), a general, simple yet effective graph neural network-based model for vulnerability detection.

Graph construction            |  Graph neural networks with residual connection
:-------------------------:|:-------------------------:
![](https://github.com/daiquocnguyen/GNN-ReGVD/blob/master/graph_construction.png)  |  ![](https://github.com/daiquocnguyen/GNN-ReGVD/blob/master/ReGVD.png)


## Usage
The repository is partially based on [CodeXGLUE](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection).

### Training and Evaluation

```shell
cd code
python run.py --output_dir=./saved_models/regcn_l2_hs128_uni_ws5_lr5e4 --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
	--do_eval --do_test --do_train --train_data_file=../dataset/train.jsonl --eval_data_file=../dataset/valid.jsonl --test_data_file=../dataset/test.jsonl \
	--block_size 400 --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
	--gnn ReGCN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
	--seed 123456 2>&1 | tee $logp/training_log.txt
```

#### Requirements
- Python 	3.7
- Pytorch 	1.9
- Transformer 	4.4

## Cite  
Please cite the paper whenever our ReGVD is used to produce published results or incorporated into other software:

	@inproceedings{NguyenReGVD,
		author={Van-Anh Nguyen and Dai Quoc Nguyen and Van Nguyen and Trung Le and Quan Hung Tran and Dinh Phung},
		title={ReGVD: Revisiting Graph Neural Networks for Vulnerability Detection},
		booktitle={Proceedings of the 44th International Conference on Software Engineering Companion (ICSE '22 Companion)},
		year={2022}
	}

## License
As a free open-source implementation, ReGVD is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. All other warranties including, but not limited to, merchantability and fitness for purpose, whether express, implied, or arising by operation of law, course of dealing, or trade usage are hereby disclaimed. I believe that the programs compute what I claim they compute, but I do not guarantee this. The programs may be poorly and inconsistently documented and may contain undocumented components, features or modifications. I make no guarantee that these programs will be suitable for any application.
