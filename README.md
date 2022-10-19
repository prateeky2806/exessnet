### Folder structure:
- data: This folder contains the final preprocessed data for the natural language domain for vision domain the datasets are downloaded automatically. This folder also contains the dataloading script.
- models: This folder contains the models including ResNet18, ResNet50, LetNeT, StaticCNN, etc. 
- trainers: contains the training scripts for the training and evaluation loops. 
- utils: contains some utilitiy functions for the project.


### Making the environment
- conda create -n cl python=3.8.1
- conda activate cl
- pip install -r requirements.txt


### Steps for running experiments.

Set the --server-home to the directory where you want to download your datasets. 
For vision datasets the --dataname flag can be splitcifar100, splitmnist, tinyimagenet
For language datasets --text-tasks can be any subset of [ag,yelp,amazon,yahoo,dbpedia] for WebNLP dataset or of [qqp,qnli,cola,mnli,sst2] for GLUE datasets.

1. Running ExSSNeT for vision dataset.
   - python main_vision.py --server-home=Add_path_your_home_directory --config=configs/transfer/config/vision.yaml --dataname=splitcifar100 --num-tasks=5 --log-dir=runs/debug --name=test --epochs=50 --weight-epochs=50 --weight-mask-type=exclusive --lr=0.01 --train-weight-lr=0.001

2. Running SSNeT for vision dataset.
   - python main_vision.py --server-home=Add_path_your_home_directory --config=configs/transfer/config/vision.yaml --dataname=splitcifar100 --num-tasks=5 --log-dir=runs/debug --name=test --epochs=50 --weight-epochs=50 --lr=0.01 --train-weight-lr=0.001

3. Running ExSSNeT for language dataset.
   - python main_text.py --config=configs/transfer/config/text.yaml --server-home=Add_path_your_home_directory --log-dir=runs/debug --name=test --epochs=2 --weight-epochs=2  --weight-mask-type=exclusive --emb-model=bert-base-uncased --lr=0.001 --train-weight-lr=0.001 --text-tasks=ag,yelp,amazon,yahoo,dbpedia

4. Running SSNeT for language dataset.
   - python main_text.py --config=configs/transfer/config/text.yaml --server-home=Add_path_your_home_directory --log-dir=runs/debug --name=test --epochs=2 --weight-epochs=2 --emb-model=bert-base-uncased --lr=0.001 --train-weight-lr=0.001 --text-tasks=ag,yelp,amazon,yahoo,dbpedia