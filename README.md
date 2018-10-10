# Correlation BKT Model

This project implement a standard Bayesian Knowledge Tracing model with correlation matrix using tensorflow. Compare to standard BKT model, this model update $p(L_t)^k_u$ for all $u$ for student $k$ on every observation using a correlation matrix.

## How to use

To train and evaluate the model, you must prepare training data and test data in following format: `[student_id labels time skill_name]`. `student_id` is a string use to identify students. Labels is observation result for this step, 0 for incorrect and 1 for correct. `time` is anything that use for sorting student sequences. `skill_name` is some text use to name the knowledge related to this step. Compositional skills are seperated by some delimiter define later. An example training data is shown beblow:

```
student1	0	step1	skillA
student1	0	step2	skillA
student1	0	step3	skillA
student1	0	step4	skillA
```

After preparing you data, you can now train and evaluate the model. Run

```python train.py -h ```

to check program usage and parameter definition. After training, all parameters and predictions are stored in the output directory you specify. `slip_param.tsv` is $p(s)^k$ for each skill. `guess_param.tsv` is $p(g)^k$ for each skill. `transit_param.tsv` is $p(t)^k$ for each skill. `corr.tsv` is the correlation matrix if you choose to train the model using correlation mode. `predictions.tsv` contains the predictions for testing data. 

Moreover, this program also store trainning details using tensorboard. There data are stored in `tensorboard/` folder.

## Example

You can train the model using toy data sets in `data/` folder, just run the following command:

```
python train.py data/toy_data.txt models --test_filename data/toy_data_test.txt
```

You should see the result come out in a few seconds.