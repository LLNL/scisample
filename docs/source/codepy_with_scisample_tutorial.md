# Using scisample with codepy

In this tutorial, you will run `codepy` with a variety of `scisample` sampling methods.

## Accessing the Model

For this tutorial, you will use an example `ares` model to set up
and run a simple workflow.  
---8<--- "docs/tutorials/.accessing-ares-model.md"
You can also
[manually set up the model](running-ares-model.md#setting-up-the-model-manually).

## Running the Model

To run the model in its default configuration, the command is `codepy run <model name>`
```
codepy run shocktube2d
```
This will create and submit a study specification.  The specification will be written
to the `codepy_study.yaml` file by default.

```yaml
---8<--- "../examples/model_files/ares/sample_output/baseline_study.yaml"
```

`codepy` will submit the study, and by default will run in the
`/p/lustre1/<username>/studies` directory.  A time-stamped directory named
`shocktube2d-study_YYYYMMDD_HHMMSS` will be created, and contain the study output.
An abbreviated directory tree is shown below.

```
studies/shocktube2d-study_YYYYMMDD_HHMMSS
`-- shocktube2d-study_YYYYMMDD-HHMMSS_1
    `-- run-shocktube2d-baseline
        |-- xxxxhsp
        `-- run-shocktube2d-baseline.slurm.sh
```

The simulation will be run in the `run-shocktube2d-baseline` directory.

---8<--- "docs/tutorials/.manual-setup-ares-model.md"
