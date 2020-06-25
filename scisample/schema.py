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

# Built-in schema
LIST_SCHEMA =  {
    'type': 'object',
    'properties': {
        'type': {'type': 'string'},
        'constants': {'type': 'object'},
        'parameters': {
            'type': 'object',
            'additionalProperties': {'type': 'array'}
        },
    },
}

# Built-in schema
COLUMN_LIST_SCHEMA =  {
    'type': 'object',
    'properties': {
        'type': {'type': 'string'},
        'constants': {'type': 'object'},
        'parameters': {'type': 'string'},
    },
}

CROSS_PRODUCT_SCHEMA =  {
    'type': 'object',
    'properties': {
        'type': {'type': 'string'},
        'constants': {'type': 'object'},
        'parameters': {
            'type': 'object',
            'additionalProperties': {'type': 'array'}
        },
    },
}

CSV_SCHEMA =  {
    'type': 'object',
    'properties': {
        'type': {'type': 'string'},
        'csv_file': {'type': 'string'},
        'row_headers': {'type': 'boolean'},
    },
    'required': ['type','csv_file','row_headers'],
}

CUSTOM_SCHEMA =  {
    'type': 'object',
    'properties': {
        'type': {'type': 'string'},
        'function': {'type': 'string'},
        'module': {'type': 'string'},
        'args': {'type': 'object'},
    },
    'required': ['type','function','module', 'args'],
}

SAMPLER_SCHEMA = {
    'list': LIST_SCHEMA,
    'column_list': COLUMN_LIST_SCHEMA,
    'cross_product': CROSS_PRODUCT_SCHEMA,
    'csv': CSV_SCHEMA,
    'custom': CUSTOM_SCHEMA,
}
