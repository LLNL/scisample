#!/usr/bin/env python3

from scisample.samplers import new_sampler

sample_dictionary = {
    'type': 'cross_product',
    'constants': {'X1': 20},
    'parameters': (
        {'X2': [5, 10],
         'X3': [5, 10]})}

sampler = new_sampler(sample_dictionary)
samples = sampler.get_samples()

print("samples:")
for sample in samples:
    print(f"    {sample}")
print()

print("parameter_block:")
for key, value in sampler.parameter_block.items():
    print(f"    {key}: {value}")
