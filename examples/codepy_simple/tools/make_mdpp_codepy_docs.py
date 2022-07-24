# use: 
# python3 tools/make_mdpp_codepy_docs.py best_candidate column_list cross_product list random uqpipeline > codepy_docs.mdpp
# markdown-pp codepy_docs.mdpp -o codepy_docs.md


import sys

print(f"There are {len(sys.argv[1:])} examples:")

for sampler in sys.argv[1:]:
	print(f"1. [{sampler}](#{sampler})")
print()

for sampler in sys.argv[1:]:
	output = (
		'''
## SAMPLER

Using 

```bash
codepy run . -c codepy_config_SAMPLER.yaml
```

to run the following `codepy_config` file

!INCLUDECODE "../../examples/codepy_simple/codepy_config_SAMPLER.yaml" (yaml), 2:999

will result in output identical (or similar) to the following: 

!INCLUDECODE "../../examples/codepy_simple/SAMPLER_out.txt" 
''')
	print(output.replace("SAMPLER", sampler))
