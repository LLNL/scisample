"""
JSON schema for validating sampler input blocks.
"""

import jsonschema


def validate_sampler(sampler_data):
    """
    Validate sampler data against the built-in schema.

    If there is no ``type`` entry in the data, it will raise a
    ``ValueError``.

    If the ``type`` entry does not match one of the built-in
    schema, it will raise a ``KeyError``.

    If the data is invalid, it will raise a ``ValidationError``.

    If no exceptions are raised, then the data is valid.

    :param sampler_data: data to validate.
    """

    if 'type' not in sampler_data:
        raise ValueError(f"No type entry in sampler data {sampler_data}")

    jsonschema.validate(
        sampler_data,
        SAMPLER_SCHEMA[sampler_data['type']]
        )


PARAMETER_SCHEMA = {
    'oneOf': [
        {
            'type': 'array'
        },
        {
            'type': 'object',
            'properties': {
                'min': {'type': 'number'},
                'max': {'type': 'number'},
                'start': {'type': 'number'},
                'stop': {'type': 'number'},
                'step': {'type': 'number'},
                'num_points': {'type': 'number'},
            },
            # Require defining each field once.
            'allOf': [
                {'oneOf': [
                    {'required': ['min']},
                    {'required': ['start']},
                ]},
                {'oneOf': [
                    {'required': ['max']},
                    {'required': ['stop']},
                ]},
                {'oneOf': [
                    {'required': ['step']},
                    {'required': ['num_points']},
                ]},
            ]
        },
        {
            'type': 'string',
            'anyOf': [
                {'pattern': (r'\[[0-9]*\.?[0-9]+:[0-9]*\.?'
                             r'[0-9]+:[0-9]*\.?[0-9]+\]')},
                {'pattern': (r'[0-9]*\.?[0-9]+ to [0-9]*\.?'
                             r'[0-9]+ by [0-9]*\.?[0-9]+')},
            ]
        }
    ]
}

# @TODO: consider moving these schema into sampling files
# Built-in schema
LIST_SCHEMA = {
    'type': 'object',
    'properties': {
        'type': {'type': 'string'},
        'constants': {'type': 'object'},
        'parameters': {
            'type': 'object',
            'additionalProperties': PARAMETER_SCHEMA
        },
    },
    'required': ['type'],
}


# Built-in schema
COLUMN_LIST_SCHEMA = {
    'type': 'object',
    'properties': {
        'type': {'type': 'string'},
        'constants': {'type': 'object'},
        'parameters': {'type': 'string'},
    },
    'required': ['type'],
}

CROSS_PRODUCT_SCHEMA = {
    'type': 'object',
    'properties': {
        'type': {'type': 'string'},
        'constants': {'type': 'object'},
        'parameters': {
            'type': 'object',
            'additionalProperties': PARAMETER_SCHEMA
        },
    },
    'required': ['type'],
}

CSV_SCHEMA = {
    'type': 'object',
    'properties': {
        'type': {'type': 'string'},
        'csv_file': {'type': 'string'},
        'row_headers': {'type': 'boolean'},
    },
    'required': ['type', 'csv_file', 'row_headers'],
}

CUSTOM_SCHEMA = {
    'type': 'object',
    'properties': {
        'type': {'type': 'string'},
        'function': {'type': 'string'},
        'module': {'type': 'string'},
        'args': {'type': 'object'},
    },
    'required': ['type', 'function', 'module', 'args'],
}

# @TODO: be more specific about parameters min/max
RANDOM_SCHEMA = {
    'type': 'object',
    'properties': {
        'type': {'type': 'string'},
        'num_samples': {'type': 'integer'},
        'previous_samples': {'type': 'string'},
        'constants': {'type': 'object'},
        'parameters': {
            'type': 'object'
        },
    },
    'required': ['type', 'num_samples'],
}

BEST_CANDIDATE_SCHEMA = RANDOM_SCHEMA


UQPIPELINE_SCHEMA = {
    'type': 'object',
    'properties': {
        'type': {'type': 'string'},
        'uq_samples': {'type': 'string'},
        'uq_points': {'type': 'string'},
        'uq_variables': {'type': 'string'},
        'uq_code': {'type': 'string'},
    },
    'required': ['type', 'uq_code'],
}
SAMPLER_SCHEMA = {
    'list': LIST_SCHEMA,
    'column_list': COLUMN_LIST_SCHEMA,
    'cross_product': CROSS_PRODUCT_SCHEMA,
    'csv': CSV_SCHEMA,
    'custom': CUSTOM_SCHEMA,
    'random': RANDOM_SCHEMA,
    'best_candidate': BEST_CANDIDATE_SCHEMA,
    'uqpipeline': UQPIPELINE_SCHEMA
}
