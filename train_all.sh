python scripts/train.py +exp=5-5_cls.yaml model=cls_wav2vec2 render_files=False logs_dir=/scratch/cjs-log
python scripts/train.py +exp=5-5_cls.yaml model=cls_panns_44k render_files=False logs_dir=/scratch/cjs-log
python scripts/train.py +exp=5-5_cls.yaml model=cls_panns_16k render_files=False logs_dir=/scratch/cjs-log 
python scripts/train.py +exp=5-5_cls.yaml model=cls_panns_pt render_files=False logs_dir=/scratch/cjs-log
python scripts/train.py +exp=5-5_cls.yaml model=cls_vggish render_files=False logs_dir=/scratch/cjs-log 
python scripts/train.py +exp=5-5_cls.yaml model=cls_wav2clip render_files=False logs_dir=/scratch/cjs-log 