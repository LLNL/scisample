There are 9 examples:
1. [best_candidate](#best_candidate)
1. [column_list](#column_list)
1. [cross_product](#cross_product)
1. [csv_column](#csv_column)
1. [csv_row](#csv_row)
1. [custom](#custom)
1. [list](#list)
1. [random](#random)
1. [uqpipeline](#uqpipeline)

## best_candidate
Using 

```bash
codepy run . -c codepy_config_best_candidate.yaml
```

to run the following `codepy_config` file

```yaml

setup:
  interactive: true
  sleep: 1
  autoyes: true
  study_name: best_candidate
  sampler:
    type: best_candidate
    num_samples: 2
    constants:
      X1: 20
      s_type: best_candidate
    parameters:
      X2:
        min: 5
        max: 10
      X3:
        min: 5
        max: 10
```

will result in output identical (or similar) to the following: 

```
::::::::::::::
best_candidate/X1.20.X2.7.602642791775893.X3.8.05887326792994.s_type.best_candidate/out.txt
::::::::::::::
{20, 7.602642791775893, 8.05887326792994}
::::::::::::::
best_candidate/X1.20.X2.9.621633469241472.X3.5.619325094643691.s_type.best_candidate/out.txt
::::::::::::::
{20, 9.621633469241472, 5.619325094643691}
```


## column_list

Using 

```bash
codepy run . -c codepy_config_column_list.yaml
```

to run the following `codepy_config` file

```yaml

setup:
  interactive: true
  sleep: 1
  autoyes: true
  study_name: column_list
  sampler:
    type: column_list
    constants:
      X1: 20
      s_type: column_list
    parameters: |
      X2  X3
      5   5
      10  10
```

will result in output identical (or similar) to the following: 

```
::::::::::::::
column_list/X1.20.X2.10.X3.10.s_type.column_list/out.txt
::::::::::::::
{20, 10, 10}
::::::::::::::
column_list/X1.20.X2.5.X3.5.s_type.column_list/out.txt
::::::::::::::
{20, 5, 5}
```


## cross_product

Using 

```bash
codepy run . -c codepy_config_cross_product.yaml
```

to run the following `codepy_config` file

```yaml

setup:
  interactive: true
  sleep: 1
  autoyes: true
  study_name: cross_product
  sampler:
    type: cross_product
    constants:
      X1: 20
      s_type: cross_product
    parameters:
      X2: [5, 10]
      X3: [5, 10]
```

will result in output identical (or similar) to the following: 

```
::::::::::::::
cross_product/X1.20.X2.10.X3.10.s_type.cross_product/out.txt
::::::::::::::
{20, 10, 10}
::::::::::::::
cross_product/X1.20.X2.10.X3.5.s_type.cross_product/out.txt
::::::::::::::
{20, 10, 5}
::::::::::::::
cross_product/X1.20.X2.5.X3.10.s_type.cross_product/out.txt
::::::::::::::
{20, 5, 10}
::::::::::::::
cross_product/X1.20.X2.5.X3.5.s_type.cross_product/out.txt
::::::::::::::
{20, 5, 5}
```


## csv_column

Using 

```bash
codepy run . -c codepy_config_csv_column.yaml
```

to run the following `codepy_config` file

```yaml

## cat column_test.csv
# X1,X2,X3
# 20,5,5
# 20,10,10

setup:
  interactive: true
  sleep: 1
  autoyes: true
  study_name: csv_column
  sampler:
    type: csv
    csv_file: column_test.csv
    row_headers: false
```

will result in output identical (or similar) to the following: 

```
::::::::::::::
csv_column/X1.20.0.X2.10.0.X3.10.0/out.txt
::::::::::::::
{20.0, 10.0, 10.0}
::::::::::::::
csv_column/X1.20.0.X2.5.0.X3.5.0/out.txt
::::::::::::::
{20.0, 5.0, 5.0}
```


## csv_row

Using 

```bash
codepy run . -c codepy_config_csv_row.yaml
```

to run the following `codepy_config` file

```yaml

## cat row_test.csv
# X1,20,20
# X2,5,10
# X3,5,10

setup:
  interactive: true
  sleep: 1
  autoyes: true
  study_name: csv_row
  sampler:
    type: csv
    csv_file: row_test.csv
    row_headers: True
```

will result in output identical (or similar) to the following: 

```
::::::::::::::
csv_row/X1.20.0.X2.10.0.X3.10.0/out.txt
::::::::::::::
{20.0, 10.0, 10.0}
::::::::::::::
csv_row/X1.20.0.X2.5.0.X3.5.0/out.txt
::::::::::::::
{20.0, 5.0, 5.0}
```


## custom

Using 

```bash
codepy run . -c codepy_config_custom.yaml
```

to run the following `codepy_config` file

```yaml

## cat custom_function.py
# def test_function(num_samples):
#     return [{"X1": 20, "X2": 5, "X3": 5},
#             {"X1": 20, "X2": 10, "X3": 10}][:num_samples]

setup:
  interactive: true
  sleep: 1
  autoyes: true
  study_name: custom
  sampler:
    type: custom
    function: test_function
    module: custom_function.py
    args:
      num_samples: 2
```

will result in output identical (or similar) to the following: 

```
::::::::::::::
custom/X1.20.X2.10.X3.10/out.txt
::::::::::::::
{20, 10, 10}
::::::::::::::
custom/X1.20.X2.5.X3.5/out.txt
::::::::::::::
{20, 5, 5}
```


## list

Using 

```bash
codepy run . -c codepy_config_list.yaml
```

to run the following `codepy_config` file

```yaml

setup:
  interactive: true
  sleep: 1
  autoyes: true
  study_name: list
  sampler:
    type: list
    constants:
      X1: 20
      s_type: list
    parameters:
      X2: [5, 10]
      X3: [5, 10]
```

will result in output identical (or similar) to the following: 

```
::::::::::::::
list/X1.20.X2.10.X3.10.s_type.list/out.txt
::::::::::::::
{20, 10, 10}
::::::::::::::
list/X1.20.X2.5.X3.5.s_type.list/out.txt
::::::::::::::
{20, 5, 5}
```


## random

Using 

```bash
codepy run . -c codepy_config_random.yaml
```

to run the following `codepy_config` file

```yaml

setup:
  interactive: true
  sleep: 1
  autoyes: true
  study_name: random
  sampler:
    type: random
    num_samples: 2
    constants:
      X1: 20
      s_type: random
    parameters:
      X2:
        min: 5
        max: 10
      X3:
        min: 5
        max: 10
```

will result in output identical (or similar) to the following: 

```
::::::::::::::
random/X1.20.X2.6.0683107773232186.X3.7.965846349547918.s_type.random/out.txt
::::::::::::::
{20, 6.0683107773232186, 7.965846349547918}
::::::::::::::
random/X1.20.X2.9.440349297581555.X3.5.801121329066036.s_type.random/out.txt
::::::::::::::
{20, 9.440349297581555, 5.801121329066036}
```


## uqpipeline

Using 

```bash
codepy run . -c codepy_config_uqpipeline.yaml
```

to run the following `codepy_config` file

```yaml

setup:
  interactive: true
  sleep: 1
  autoyes: true
  study_name: uqpipeline
  sampler:
    type: uqpipeline
    uq_points: points
    uq_variables:
      - X1
      - X2
    uq_code: |
      points = sampler.CartesianCrossSampler.sample_points(
        num_divisions=[3,3], 
        box=[[-1,1],[]], 
        values=[[],['foo', 'bar']])    
```

will result in output identical (or similar) to the following: 

```
::::::::::::::
uqpipeline/X1.-1.0.X2.bar/out.txt
::::::::::::::
{-1.0, bar, }
::::::::::::::
uqpipeline/X1.-1.0.X2.foo/out.txt
::::::::::::::
{-1.0, foo, }
::::::::::::::
uqpipeline/X1.0.0.X2.bar/out.txt
::::::::::::::
{0.0, bar, }
::::::::::::::
uqpipeline/X1.0.0.X2.foo/out.txt
::::::::::::::
{0.0, foo, }
::::::::::::::
uqpipeline/X1.1.0.X2.bar/out.txt
::::::::::::::
{1.0, bar, }
::::::::::::::
uqpipeline/X1.1.0.X2.foo/out.txt
::::::::::::::
{1.0, foo, }
```

