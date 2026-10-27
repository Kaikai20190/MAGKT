# Data

The datasets are not included in this repository.

| dataset | source | file(s) |
|---|---|---|
| ASSISTments2009 | [ASSISTments 2009-2010 skill-builder data](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010) | `skill_builder_data.csv` |
| Statics2011 | [PSLC DataShop, dataset 507](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=507) | `AllData_student_step_2011F.csv` |
| KDDCup2010 | [KDD Cup 2010, Algebra I 2005-2006](https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp) | `algebra_2005_2006_train.txt` |
| ASSISTments2017 | preprocessed version from the [DGEKT repository](https://github.com/Yumo216/DGEKT) | `assist2017_train.csv`, `assist2017_test.csv`, `2017.csv` |

Put the raw files of the first three datasets in `data/raw/` and run `step1_prepare_data.py`, which writes the
sequence files and hypergraphs to `data/processed/`. Put the ASSISTments2017 files in `data/assist2017/`
(sequence files) and `data/H/` (hypergraph). The file locations can be changed in `configs/datasets/*.yaml`.
