
## best_candidate {#best_candidate}

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
  sampler:
    type: best_candidate
    num_samples: 5
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

will result in the following `out.txt` files.

```
::::::::::::::
X1.20.X2.5.161752200752359.X3.9.921390919925802.s_type.best_candidate/out.txt
::::::::::::::
{20, 5.161752200752359, 9.921390919925802}
::::::::::::::
X1.20.X2.5.265568649708979.X3.6.052470560491571.s_type.best_candidate/out.txt
::::::::::::::
{20, 5.265568649708979, 6.052470560491571}

```


## column_list {#column_list}

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
  sampler:
    type: column_list
    constants:
      X1: 20
      s_type: column_list
    parameters: "X2  X3\n5   5\n10  10 \n"

```

will result in the following `out.txt` files.

```
::::::::::::::
X1.20.X2.10.X3.10.s_type.column_list/out.txt
::::::::::::::
{20, 10, 10}
::::::::::::::
X1.20.X2.5.X3.5.s_type.column_list/out.txt
::::::::::::::
{20, 5, 5}

```


## cross_product {#cross_product}

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
  sampler:
    type: cross_product
    constants:
      X1: 20
      s_type: cross_product
    parameters:
      X2: [5, 10]
      X3: [5, 10]

```

will result in the following `out.txt` files.

```
::::::::::::::
X1.20.X2.10.X3.10.s_type.cross_product/out.txt
::::::::::::::
{20, 10, 10}
::::::::::::::
X1.20.X2.10.X3.5.s_type.cross_product/out.txt
::::::::::::::
{20, 10, 5}
::::::::::::::
X1.20.X2.5.X3.10.s_type.cross_product/out.txt
::::::::::::::
{20, 5, 10}
::::::::::::::
X1.20.X2.5.X3.5.s_type.cross_product/out.txt
::::::::::::::
{20, 5, 5}

```


## list {#list}

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
  sampler:
    type: list
    constants:
      X1: 20
      s_type: list
    parameters:
      X2: [5, 10]
      X3: [5, 10]

```

will result in the following `out.txt` files.

```
::::::::::::::
X1.20.X2.10.X3.10.s_type.list/out.txt
::::::::::::::
{20, 10, 10}
::::::::::::::
X1.20.X2.5.X3.5.s_type.list/out.txt
::::::::::::::
{20, 5, 5}

```


## random {#random}

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
  sampler:
    type: random
    num_samples: 5
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

will result in the following `out.txt` files.

```
::::::::::::::
X1.20.X2.5.216116379276561.X3.9.596853343739944.s_type.random/out.txt
::::::::::::::
{20, 5.216116379276561, 9.596853343739944}
::::::::::::::
X1.20.X2.9.290368859808531.X3.8.759626617749904.s_type.random/out.txt
::::::::::::::
{20, 9.290368859808531, 8.759626617749904}

```


## uqpipeline {#uqpipeline}

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
  sampler:
    type: uqpipeline
    uq_points: points
    uq_variables:
    - X1
    - X2
    uq_code: "points = sampler.CartesianCrossSampler.sample_points(\n    num_divisions=[3,3],\
      \ \n    box=[[-1,1],[]], \n    values=[[],['foo', 'bar']])\n"

```

will result in the following `out.txt` files.

```
::::::::::::::
X1.-1.0.X2.bar/out.txt
::::::::::::::
{-1.0, bar, }
::::::::::::::
X1.-1.0.X2.foo/out.txt
::::::::::::::
{-1.0, foo, }
::::::::::::::
X1.0.0.X2.bar/out.txt
::::::::::::::
{0.0, bar, }
::::::::::::::
X1.0.0.X2.foo/out.txt
::::::::::::::
{0.0, foo, }
::::::::::::::
X1.1.0.X2.foo/out.txt
::::::::::::::
{1.0, foo, }

```

