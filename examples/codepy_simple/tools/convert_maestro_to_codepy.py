#!/usr/bin/env python3
"""Converts maestro examples to codepy examples.
   Usage: ./tools/convert_maestro_to_codepy.py ../maestrowf/sample*yaml 
"""

import sys
import yaml

filenames = sys.argv[1:]
for filename in filenames:
    try:
        new_filename = filename.replace('sample_', 'codepy_config_')
        new_filename = new_filename.split('/')[-1]
        print(f'converting {filename} to {new_filename}')
        print()
        with open(filename) as file:
            old_dict = yaml.safe_load(file)
        new_dict = {
            'setup': {
                'interactive': True,
                'sleep': 1,
                'autoyes': True,
                'sampler': old_dict['env']['variables']['SAMPLE_DICTIONARY']
            }
        }
        if ('parameters' in new_dict['setup']['sampler']
            and type(new_dict['setup']['sampler']['parameters']) == dict):
            for parameter in new_dict['setup']['sampler']['parameters']:
                new_dict['setup']['sampler']['parameters'][parameter] = str(
                    new_dict['setup']['sampler']['parameters'][parameter])
        if 'constants' in new_dict['setup']['sampler']:
            new_dict['setup']['sampler']['constants']['s_type'] = (
                new_dict['setup']['sampler']['type'])
        new_dict_string = yaml.dump(new_dict, sort_keys=False)
        new_dict_string = new_dict_string.replace("'[", "[")
        new_dict_string = new_dict_string.replace("]'", "]")
        with open(new_filename, 'w') as file:
            file.write(f'# run this with codepy run . -c {new_filename}\n\n')
            file.write(new_dict_string)
    except (yaml.scanner.ScannerError) as e:
        print(f'ERROR: converting {filename} to {new_filename}')
        print(f'ERROR: {e}')
