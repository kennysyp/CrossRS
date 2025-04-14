```
CrossRS/
├── CrossRS-LPBA/
│   └── data/
│   │   └── data_utils.py
│   │   └── labels.txt
│   │   └── test.csv
│   │   └── train.csv
│   │   └── val.csv
│   └── model/
│   └── networks/
│   │   └── models.py
│   │   └── symnet.py
│   └── utils/
│   │   └── util.py
│   └── train_loop.py
│   └── test_reg.py
│   └── test_seg.py
├── CrossRS-OASIS/
│   └── data/
│   │   └── data_utils.py
│   │   └── seg35_labels.txt
│   │   └── test.csv
│   │   └── train.csv
│   │   └── val.csv
│   └── model/
│   └── networks/
│   │   └── models.py
│   │   └── symnet.py
│   └── utils/
│   │   └── util.py
│   └── train_loop.py
│   └── test_reg.py
│   └── test_seg.py
└── requirements.txt
```

#### train OASIS

```bash
cd Cross-OASIS
python train_loop.py
```

#### test OASIS

```bash
python test_seg.py
python test_reg.py
```

#### train LPBA

```bash
cd Cross-LPBA
python train_loop.py
```

#### test LPBA

```bash
python test_seg.py
python test_reg.py
```

Python 3.9.13
