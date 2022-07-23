
* update docstring for parameter_block:     

  self.get_samples: [
    {'X1': 20, 'X2': 5, 'X3': 5, 'X4': 5}, 
    {'X1': 20, 'X2': 10, 'X3': 10, 'X4': 10}]   

 self._parameter_block: {
    'X1': {'values': [20, 20], 'label': 'X1.%%'}, 
    'X2': {'values': [5, 10], 'label': 'X2.%%'}, 
    'X3': {'values': [5, 10], 'label': 'X3.%%'}, 
    'X4': {'values': [5, 10], 'label': 'X4.%%'}}       

* potential improvements:
1. Support previous_samples for `best_candidate`
1. To support list comprehension using `eval`.
1. Merge in composite sampler from daub1
1. Add webhook for read the docs
1. Consider merging random tests into best candidate tests

