"""
Sampler Tests

Unit tests for the codepy sampler objects:
#. ListSampler,
#. ColumnListSampler,
#. CrossProductSampler,
#. RandomSampler,
#. BestCandidateSampler,
#. CsvSampler,
#. CustomSampler

More tests on invalid samplers are likely needed.
"""

import os
import shutil
import tempfile
import unittest
from contextlib import suppress

import pytest
import yaml

from scisample.best_candidate_sampler import BestCandidateSampler
from scisample.column_list_sampler import ColumnListSampler
from scisample.cross_product_sampler import CrossProductSampler
from scisample.list_sampler import ListSampler
from scisample.random_sampler import RandomSampler
from scisample.custom_sampler import CustomSampler
from scisample.csv_sampler import CsvSampler
from scisample.samplers import new_sampler
from scisample.utils import SamplingError, read_yaml #, new_sampler_from_yaml

PANDAS_PLUS = False
with suppress(ModuleNotFoundError):
    import pandas as pd
    import numpy as np
    import scipy.spatial as spatial
    PANDAS_PLUS = True

# @TODO: improve coverage

def new_sampler_from_yaml(yaml_text):
    """Returns sampler from yaml text"""
    return new_sampler(
        yaml.safe_load(yaml_text))

class TestScisampleExceptions(unittest.TestCase):
    """
    Scenario: Requesting samplers with invalid yaml input
    """

    def test_missing_type_exception(self):
        """
        Given a missing sampler type,
        And I request a new sampler
        Then I should get a SamplerException
        """
        yaml_text = """
            foo: bar
            #constants:
            #    X1: 20
            #parameters:
            #   X2: [5, 10]
            #   X3: [5, 10]
            """
        with self.assertRaises(SamplingError) as context:
            new_sampler_from_yaml(yaml_text)
        self.assertTrue(
            "No type entry in sampler data"
            in str(context.exception))

    def test_invalid_type_exception(self):
        """
        Given an invalid sampler type,
        And I request a new sampler
        Then I should get a SamplerException
        """
        yaml_text = """
            type: foobar
            #constants:
            #    X1: 20
            #parameters:
            #   X2: [5, 10]
            #   X3: [5, 10]
            """
        with self.assertRaises(SamplingError) as context:
            new_sampler_from_yaml(yaml_text)
        self.assertTrue(
            "not a recognized sampler type"
            in str(context.exception))

    def test_missing_data_exception(self):
        """
        Given no constants or parameters
        And I request a new sampler
        Then I should get a SamplerException
        """
        yaml_text = """
            type: list
            #constants:
            #    X1: 20
            #parameters:
            #   X2: [5, 10]
            #   X3: [5, 10]
            """
        with self.assertRaises(SamplingError) as context:
            new_sampler_from_yaml(yaml_text)
        self.assertTrue(
            "Either constants or parameters must be included"
            in str(context.exception))

    def test_duplicate_data_exception(self):
        """
        Given a variable in both constants and parameters
        And I request a new sampler
        Then I should get a SamplerException
        """
        # @TODO: We can not detect if parameters are defined twice.
        # @TODO: Fixing this requires a rewrite of read_yaml.
        yaml_text = """
            type: list
            constants:
                X2: 20
            parameters:
                X2: [5, 10]
                X2: [5, 10]
                X3: [5, 10]
                X3: [5, 10]
             """
        with self.assertRaises(SamplingError) as context:
            new_sampler_from_yaml(yaml_text)
        self.assertTrue(
            "The following constants or parameters are defined more than once"
            in str(context.exception))


class TestScisampleUniversal(unittest.TestCase):
    """
    Scenario: Testing behavior valid for multiple samplers
    """
    def test_constants_only(self):
        """
        Given only constants
        And I request a new sampler
        Then I should get a sampler with one sample
        With appropriate values
        """
        yaml_text = """
            type: list
            constants:
                X1: 20
                X2: 30
            #parameters:
            #    X2: [5, 10]
            #    X3: [5, 10]
            """
        sampler = new_sampler_from_yaml(yaml_text)
        samples = sampler.get_samples()

        self.assertEqual(len(samples), 1)
        for sample in samples:
            self.assertEqual(sample['X1'], 20)
            self.assertEqual(sample['X2'], 30)

    def test_parameters_only(self):
        """
        Given only parameters
        And I request a new sampler
        Then I should get appropriate values
        """
        yaml_text = """
            type: list
            #constants:
            #    X1: 20
            parameters:
                X2: [5, 10]
                X3: [5, 10]
            """
        sampler = new_sampler_from_yaml(yaml_text)
        samples = sampler.get_samples()

        self.assertEqual(len(samples), 2)

        self.assertEqual(samples[0]['X2'], 5)
        self.assertEqual(samples[0]['X3'], 5)
        self.assertEqual(samples[1]['X2'], 10)
        self.assertEqual(samples[1]['X3'], 10)


class TestScisampleList(unittest.TestCase):
    """
    Scenario: normal and abnormal tests for ListSampler
    """

    def test_normal(self):
        """
        Given a list specification
        And I request a new sampler
        Then I should get a ListSampler
        With appropriate values
        """
        yaml_text = """
            type: list
            constants:
                X1: 20
            parameters:
                X2: [5, 10]
                X3:
                    min: 5
                    max: 10
                    step: 5
                X4: 5.0 to 10 by 5.0
                X5: "[5.0:10.0:5]"
                X6:
                    start: 5
                    stop: 10
                    num_points: 2
            """
        sampler = new_sampler_from_yaml(yaml_text)
        self.assertTrue(isinstance(sampler, ListSampler))

        samples = sampler.get_samples()

        self.assertEqual(len(samples), 2)
        for sample in samples:
            self.assertEqual(sample['X1'], 20)
        for i in range(2, 7):
            self.assertEqual(samples[0][f'X{i}'], 5)
            self.assertEqual(samples[1][f'X{i}'], 10)
        sampler
        self.assertEqual(samples,
            [{'X1': 20, 'X2': 5, 'X3': 5, 'X4': 5, 'X5': 5, 'X6': 5},
             {'X1': 20, 'X2': 10, 'X3': 10, 'X4': 10, 'X5': 10, 'X6': 10}])
        self.assertEqual(sampler.parameter_block,
            {'X1': {'values': [20, 20], 'label': 'X1.%%'},
             'X2': {'values': [5, 10], 'label': 'X2.%%'},
             'X3': {'values': [5, 10], 'label': 'X3.%%'},
             'X4': {'values': [5, 10], 'label': 'X4.%%'},
             'X5': {'values': [5, 10], 'label': 'X5.%%'},
             'X6': {'values': [5, 10], 'label': 'X6.%%'}})

    def test_error(self):
        """
        Given an invalid list specification
        And I request a new sampler
        Then I should get a SamplerException
        """
        yaml_text = """
            type: list
            constants:
                X1: 20
            parameters:
                X2: [5, 10, 20]
                X3: [5, 10]
                X4: [5, 10]
            """
        with self.assertRaises(SamplingError) as context:
            new_sampler_from_yaml(yaml_text)
        self.assertTrue(
            "All parameters must have the same number of values"
            in str(context.exception))


class TestScisampleCrossProduct(unittest.TestCase):
    """
    Scenario: normal tests for CrossProductSampler
    """
    def test_normal(self):
        """
        Given a cross_product specification
        And I request a new sampler
        Then I should get a CrossProductSampler
        With appropriate values
        """
        yaml_text = """
            # sampler:
                type: cross_product
                constants:
                    X1: 20
                parameters:
                    X2: [5, 10]
                    X3: [5, 10]
            """
        sampler = new_sampler_from_yaml(yaml_text)
        self.assertTrue(isinstance(sampler, CrossProductSampler))

        samples = sampler.get_samples()

        self.assertEqual(sampler.parameters, ["X1", "X2", "X3"])
        self.assertEqual(len(samples), 4)

        for sample in samples:
            self.assertEqual(sample['X1'], 20)
        self.assertEqual(samples[0]['X2'], 5)
        self.assertEqual(samples[0]['X3'], 5)
        self.assertEqual(samples[1]['X2'], 5)
        self.assertEqual(samples[1]['X3'], 10)
        self.assertEqual(samples[2]['X2'], 10)
        self.assertEqual(samples[2]['X3'], 5)
        self.assertEqual(samples[3]['X2'], 10)
        self.assertEqual(samples[3]['X3'], 10)


class TestScisampleColumnList(unittest.TestCase):
    """
    Scenario: normal and abnormal tests for ColumnListSampler
    """
    def test_normal(self):
        """
        Given a column_list specification
        And I request a new sampler
        Then I should get a ColumnListSampler
        With appropriate values
        """
        yaml_text = """
            type: column_list
            constants:
                X1: 20
            parameters: |
                X2     X3     X4
                5      5      5
                10     10     10
            """
        sampler = new_sampler_from_yaml(yaml_text)
        self.assertTrue(isinstance(sampler, ColumnListSampler))

        samples = sampler.get_samples()

        self.assertEqual(len(samples), 2)
        for sample in samples:
            self.assertEqual(sample['X1'], 20)
        self.assertEqual(samples[0]['X2'], '5')
        self.assertEqual(samples[0]['X3'], '5')
        self.assertEqual(samples[0]['X4'], '5')
        self.assertEqual(samples[1]['X2'], '10')
        self.assertEqual(samples[1]['X3'], '10')
        self.assertEqual(samples[1]['X4'], '10')

    def test_comments(self):
        """
        Given a column_list specification
        And I request a new sampler
        Then I should get a ColumnListSampler
        With appropriate values
        And any commented lines should be ignored.
        """
        yaml_text = """
            type: column_list
            constants:
                X1: 20
            parameters: |
                X2     X3     X4
                5      5      5 # This is a comment
                10     10     10
                #15    15     15 # Don't process this line
            """
        sampler = new_sampler_from_yaml(yaml_text)
        self.assertTrue(isinstance(sampler, ColumnListSampler))

        samples = sampler.get_samples()

        self.assertEqual(len(samples), 2)
        for sample in samples:
            self.assertEqual(sample['X1'], 20)
        self.assertEqual(samples[0]['X2'], '5')
        self.assertEqual(samples[0]['X3'], '5')
        self.assertEqual(samples[0]['X4'], '5')
        self.assertEqual(samples[1]['X2'], '10')
        self.assertEqual(samples[1]['X3'], '10')
        self.assertEqual(samples[1]['X4'], '10')

    def test_error(self):
        """
        Given an invalid column_list specification
        And I request a new sampler
        Then I should get a SamplerException
        """
        yaml_text = """
            type: column_list
            constants:
                X1: 20
            parameters: |
                X2     X3     X4
                5      5      5
                10     10     10
                20
            """
        with self.assertRaises(SamplingError) as context:
            new_sampler_from_yaml(yaml_text)
        self.assertTrue(
            "All rows must have the same number of values"
            in str(context.exception))

class TestScisampleRandomSampler(unittest.TestCase):
    """
    Scenario: normal and abnormal tests for RandomSampler
    """
    def test_normal(self):
        """
        Given a random specification
        And I request a new sampler
        Then I should get a RandomSampler
        With appropriate values
        """
        yaml_text = """
            type: random
            num_samples: 5
            #previous_samples: samples.csv # optional
            constants:
                X1: 20
                X2: foo
            parameters:
                X3:
                    min: 5
                    max: 10
                X4:
                    min: 5
                    max: 10
            """
        sampler = new_sampler_from_yaml(yaml_text)
        self.assertTrue(isinstance(sampler, RandomSampler))

        samples = sampler.get_samples()

        self.assertEqual(len(samples), 5)
        for sample in samples:
            self.assertEqual(sample['X1'], 20)
            self.assertEqual(sample['X2'], "foo")
            self.assertTrue(sample['X3'] > 5)
            self.assertTrue(sample['X4'] > 5)
            self.assertTrue(sample['X3'] < 10)
            self.assertTrue(sample['X4'] < 10)
    def test_normal2(self):
        """
        Given a random specification
        And I request a new sampler
        Then I should get a RandomSampler
        With appropriate values
        """
        yaml_text = """
            type: random
            num_samples: 5
            #previous_samples: samples.csv # optional
            constants:
                X1: 0.5
            parameters:
                X2:
                    min: 0.2
                    max: 0.8
                X3:
                    min: 0.2
                    max: 0.8
            """
        sampler = new_sampler_from_yaml(yaml_text)
        self.assertTrue(isinstance(sampler, RandomSampler))

        samples = sampler.get_samples()

        self.assertEqual(len(samples), 5)
        for sample in samples:
            self.assertEqual(sample['X1'], 0.5)
            self.assertTrue(sample['X2'] > 0.2)
            self.assertTrue(sample['X3'] > 0.2)
            self.assertTrue(sample['X2'] < 0.8)
            self.assertTrue(sample['X3'] < 0.8)

    def test_error1(self):
        """
        Given an invalid random specification
        And I request a new sampler
        Then I should get a SamplerException
        """
        yaml_text = """
            type: random
            num_samples: 5
            #previous_samples: samples.csv # optional
            constants:
                X1: 20
            parameters:
                X2:
                    min: foo
                    max: 10
                X3:
                    min: 5
                    max: 10
            """
        with self.assertRaises(SamplingError) as context:
            new_sampler_from_yaml(yaml_text)
        self.assertTrue(
            "must have a numeric minimum"
            in str(context.exception))

    def test_error2(self):
        """
        Given an invalid random specification
        And I request a new sampler
        Then I should get a SamplerException
        """
        yaml_text = """
            type: random
            num_samples: 5
            #previous_samples: samples.csv # optional
            constants:
                X1: 20
            parameters:
                X2:
                    min: 1
                    max: bar
                X3:
                    min: 5
                    max: 10
            """
        with self.assertRaises(SamplingError) as context:
            new_sampler_from_yaml(yaml_text)
        self.assertTrue(
            "must have a numeric maximum"
            in str(context.exception))

class TestScisampleBestCandidate(unittest.TestCase):
    """
    Scenario: normal and abnormal tests for BestCandidate
    """
    def test_normal(self):
        """
        Given a best_candidate specification
        And I request a new sampler
        Then I should get a BestCandidate
        With appropriate values
        """
        yaml_text = """
            type: best_candidate
            num_samples: 5
            constants:
                X1: 20
                X2: foo
            parameters:
                X3:
                    min: 5
                    max: 10
                X4:
                    min: 5
                    max: 10
            """
        if PANDAS_PLUS:
            sampler = new_sampler_from_yaml(yaml_text)
            self.assertTrue(isinstance(sampler, BestCandidateSampler))
            samples = sampler.get_samples()
            self.assertEqual(len(samples), 5)
            for sample in samples:
                self.assertEqual(sample['X1'], 20)
                self.assertEqual(sample['X2'], "foo")
                self.assertTrue(sample['X3'] > 5)
                self.assertTrue(sample['X4'] > 5)
                self.assertTrue(sample['X3'] < 10)
                self.assertTrue(sample['X4'] < 10)
        else:
            # test only works if pandas is installed
            self.assertTrue(True)

    def test_error1(self):
        """
        Given an invalid best_candidate specification
        And I request a new sampler
        Then I should get a SamplerException
        """
        yaml_text = """
            type: best_candidate
            num_samples: 5
            previous_samples: samples.csv # optional
            # missing ["cost_variable", "downselect_ratio", "voxel_overlap"]
            constants:
                X1: 20
            parameters:
                X2:
                    min: 5
                    max: 10
                X3:
                    min: 5
                    max: 10
            """
        with self.assertRaises(SamplingError) as context:
            new_sampler_from_yaml(yaml_text)
        self.assertTrue(
            "sampling requires"
            in str(context.exception))

    def test_error2(self):
        """
        Given an invalid best_candidate specification
        And I request a new sampler
        Then I should get a SamplerException
        """
        yaml_text = """
            type: best_candidate
            num_samples: 30
            previous_samples: samples.csv
            cost_variable: cost
            downselect_ratio: 1.2
            voxel_overlap: 1.0 # voxel just touches nearest neighbor
            constants:
                X1: 20
            parameters:
                X2:
                    min: 5
                    max: 10
                X3:
                    min: 5
                    max: 10
        """
        with self.assertRaises(SamplingError) as context:
            new_sampler_from_yaml(yaml_text)
        self.assertTrue(
            "The 'downselect_ratio' must be less than or equal to 1.0"
            in str(context.exception))

    def test_error2(self):
        """
        Given an invalid best_candidate specification
        And I request a new sampler
        Then I should get a SamplerException
        """
        yaml_text = """
            type: best_candidate
            num_samples: 30
            previous_samples: samples.csv
            cost_variable: cost
            downselect_ratio: 0.3
            voxel_overlap: -2.3
            constants:
                X1: 20
            parameters:
                X2:
                    min: 5
                    max: 10
                X3:
                    min: 5
                    max: 10
        """
        with self.assertRaises(SamplingError) as context:
            new_sampler_from_yaml(yaml_text)
        self.assertTrue(
            "The 'voxel_overlap' must be greater than 0.0"
            in str(context.exception))

class TestScisampleBestCandidateResample(unittest.TestCase):
    """
    Scenario: resample tests for BestCandidate
    """
# class TestCsvRowSampler(unittest.TestCase):
    # """Unit test for testing the csv sampler."""
    # CSV_SAMPLER = """
    # sampler:
    #     type: csv
    #     csv_file: {path}/test.csv
    #     row_headers: True
    # """
    RESAMPLE_SAMPLER = """
    sampler:
        type: best_candidate
        num_samples: 30
        previous_samples: samples.csv
        cost_variable: cost
        downselect_ratio: 0.3
        voxel_overlap: 1.0 # voxel just touches nearest neighbor
        constants:
            X1: 20
        parameters:
            X2:
                min: 5
                max: 10
            X3:
                min: 5
                max: 10
    """
    # Note: the csv reader does not ignore blank lines
    CSV1 = """,X1,X2,cost
    0,1.0381277436778031,-0.7096193757574083,319.4558018555205
    1,-1.9927519079161806,2.9587244608100054,111.43892208454632
    2,1.9460293475881927,2.9749499087596765,66.84241498888628
    3,-1.9913588423049453,-0.8436370176598333,2321.7377688141605
    4,-0.030239465027585233,1.5025671804925977,226.5574930895241
    5,1.8570043634932447,1.0732030753969184,564.9214754024089
    6,-1.880450915443355,1.048175415088719,627.2717047113881
    7,-0.5190284759688146,-0.12094474064155047,17.543612115008894
    8,-0.4359715978954104,2.998076829930714,790.5515570114809
    9,0.7392508345515183,2.5210971958293653,389.97463847872035
    10,-1.1381207364313393,2.006721235153301,55.180901236070575
    11,0.7769782940157599,0.5268473119075776,0.6402995379799649
    12,1.9726900997324663,-0.039302081148331336,1546.0715236011652
    13,-1.5337925743588774,0.08862484709232898,518.9420777254159
    14,-0.8768695613932427,0.9370610424280228,6.350445311551714
    15,-0.05978252284751928,-0.9989155184971961,101.62165244813387
    16,0.9693269097571089,1.50258385669434,31.696624625449108
    17,-1.029743222479011,-0.9989072922096884,428.18260895824005
    18,1.7734307447481386,2.026495552265458,125.71607830359228
    19,1.9467901264287693,-0.968278934668171,2265.0104465199565
    20,-1.9889662514473918,1.9114015342304587,426.9667894599257
    21,0.3052338424319587,-0.16216850922284998,7.002357914489589
    22,-0.07101344047865288,0.6938000256657966,48.58570640284774
    23,-0.04916754657751232,2.2936170331196624,526.0603065926351
    24,-1.2206789767758752,2.794820330764298,175.17210716297245
    25,1.2291735564946342,0.013495096178711918,224.26497161789175
    26,0.23977488521953205,2.960116723288838,843.1009732146393
    27,1.2821000857717761,2.9723294223845667,176.58376985004804
    28,0.4909000968323478,1.926140958647097,284.23494926497744
    29,-1.3226325085759312,1.3949814855157996,17.952804772047948
    30,1.4091383474552224,0.6172259794541151,187.43153939131722
    31,0.48039811183294834,1.0808266744047321,72.52752217254077
    32,-0.6594473308401025,2.423845290503807,398.35572493110953
    33,-1.148077331297349,-0.39086339628487865,296.6635221171575
    34,-0.5518125197033155,1.8171108779869947,231.2081792702479
    35,1.1922565378139938,2.1343635487592,50.857877916580776
    36,-1.6556453870940486,2.395227623614129,19.019487331338418
    37,1.5670859731233047,-0.4772238252250838,860.5600875133215
    38,-0.9024934629742289,0.3106685705746335,29.003533126199873
    39,-1.9773717257452148,0.4554938112700331,1202.2253120628925
    40,-1.3878504887777043,0.6886436028279079,158.83883563036065
    41,-0.5206968595650627,-0.6660008992038744,90.13305518580754
    42,0.48321275720438495,-0.952913521023691,141.02348459644523
    43,1.5086483832939348,1.534181066502018,55.291215256206556
    44,1.9452011941740257,0.5451065158570807,1049.8119321268316
    45,1.6401942762566524,2.534434358623453,2.8373032395250433
    46,-1.9019843850169287,-0.2755936997850177,1524.0740961566312
    47,1.0316312877748701,0.9644103150127052,0.9980586832492135
    48,-0.46638475016004266,1.2333646115670334,105.3453813706821
    49,0.3170159347250783,0.3397649786659236,6.191283165476911
    50,0.7999584209695993,-0.2825962813351728,85.14613181741822
    51,-0.16652579174448734,0.21333093367256728,4.805521925317006
    52,-0.07390640144633176,-0.4621653344187884,23.020821954948048
    53,-1.490743003906224,-0.7275122293573371,876.3516942078113
    54,1.9866604464579432,1.5633657822255174,569.0587707241955
    55,1.4287949915481093,-0.9854236184296914,916.3833596266488
    56,0.764885507977362,2.9842062872054345,575.6504444905345
    57,-1.7866318779056773,1.501540798178158,293.54862561797336
    58,-0.5288248868586782,0.6621492149392085,16.967429768289264
    59,-0.8775389870373123,1.3973574969549705,42.87352667407656
    60,1.9973293821489753,-0.5121990480214742,2027.3662360007274
    61,0.4209126471730009,1.5033885131830567,176.22157126784575
    62,0.3123264017382672,2.539635404355863,596.8520908837285
    63,1.596390632924137,0.2410178525461788,532.7860170450566
    64,-1.553945369424627,1.9734982303739117,25.99261500445374
    65,0.6599950814781255,0.1093673876338066,10.757951477828367
    66,1.4092076973220324,1.1260575443563363,74.09456644735684
    67,1.2209155920199914,2.5376668581757706,109.67639943686815
    68,0.07732669022306471,1.087632796495595,117.84872936948756
    69,-0.15503422055084437,1.9065866230694652,355.7339359040444
    70,-0.17351966854892442,2.7005262293870267,714.4899260575332
    71,-1.586906009991964,2.941408544025169,24.5966475167725
    72,0.47647228674964115,-0.5692649719917213,63.68198699922381
    73,-1.2165866512852634,2.379633172101561,85.83229318726801
    74,1.9983047024454708,2.4112520709958116,251.25939786746966
    75,1.187817594116504,-0.35519733078602966,311.9490107984959
    76,-0.819034192890856,2.7575393233390235,438.7498870590395
    77,-0.9975997429899746,-0.04345712708279903,111.87235751111002
    78,1.1034247526985537,0.3523579261572358,74.8657689805437
    79,-0.5461158440863958,0.2577461778044903,2.554469537273904
    80,1.5884646055035567,-0.12000306470931577,699.0090033970732
    81,0.2936847211178151,0.6965697390217773,37.74781233044327
    82,-1.512347949302049,1.0231401925601022,166.09568148904557
    83,0.845299560143391,1.8986724048617183,140.2429368646414
    84,-1.5633331922337868,-0.37327197413599267,800.2788267155228
    85,-1.2834885282936472,0.3350986896554935,177.41278096573302
    86,1.434128143224651,1.8807913305268387,3.283681167722275
    87,-0.7701079357496012,2.090977507402496,227.5071007909377
    88,-1.881255923785027,0.11653839390681764,1179.7107566557643
    89,-0.8042289986643403,-0.43519699743238194,120.32359124716253
    90,0.8498541864848836,-0.9997893193727223,296.56522195481364
    91,-1.7293139264424813,0.7074490882381395,528.6934726366175
    92,-0.8862402994799723,1.752774434288714,97.1350011362059
    93,0.18121761093680666,-0.7328422426188719,59.29730708719943
    94,-0.3137092828324297,0.9234481694909742,69.79405033359605
    95,-0.18735024289944402,-0.13826804047845842,4.4154522812053125
    96,-1.8669970542545893,2.650601087645972,77.95501716320476
    97,1.7040343108661604,0.7796416178231143,451.67205566580685
    98,0.7207403294726302,0.8600812651676191,11.6798194416131
    99,-1.1888478145685033,1.062096528448496,17.129596013557336
    100,0.224385208448997,2.116786014851297,427.6178869282993
    101,-1.0194749223690396,0.6457393874293853,19.569566511138067
    102,-1.2979049201680728,1.7183474736891613,5.3945454047518195
    103,0.21679166754367252,1.7507410504285734,290.8872397914042
    104,-0.3851912006812519,2.1215633475484923,391.26706103342207
    105,0.8428281137050355,1.215056561350007,25.49664270456768
    106,0.835654726486859,2.2110253700015496,228.85511943590896
    107,0.5307936428877289,2.7803643233062987,624.5315603630569
    108,1.5784599693001709,2.8827353675980905,15.638320261777599
    109,1.7247709544513885,-0.7561264946679387,1392.5325449842444
    110,-1.6964919511469758,-0.9551092609550107,1476.6088473010668
    111,-0.9809491004843451,-0.6857673981795207,275.52396485955836
    112,-0.7393262843318484,-0.8795075594714987,206.40448987639735
    113,1.4791550565384268,2.2347024271717117,0.4486392704768335
    114,-0.4100339607978247,1.5501142732298754,192.97684344408356
    115,-0.22579171712430135,-0.7277725100912504,62.14840838296951
    116,0.9873971738089149,2.7254808606280654,306.4348752998159
    117,-0.09211898025557819,2.9881565483361157,889.0364372351376
    118,1.2197160074614302,1.3498284938870135,1.9493271978287137
    119,-1.9678874326807274,2.203453559847613,287.4069795018737
    120,-0.5262743772850538,2.719581637885362,598.9672541374933
    121,0.4872153481143875,2.242389573051935,402.2697699339233
    122,-0.4097196620875079,-0.3994673531124815,34.174499613168166
    123,-1.2788540729197786,-0.10154784974615616,306.9154917264673
    124,1.7158589398596664,1.7322371325354293,147.3910424434102
    125,1.0684722117825878,0.6368541016357274,25.484848675547873
    126,-0.9416391373536648,2.4701699437532287,254.51265202257886
    127,0.7008659274602822,-0.7442383523964238,152.7234975353936
    128,-1.9545806459694535,-0.5532240830716479,1921.5756267052782
    129,-0.28566363452167076,0.46743716745340924,16.53967650904251
    130,0.9542950652135156,-0.04226838741385608,90.81297488436091
    131,1.1893646562124314,1.7487383557920984,11.201485924164814
    132,0.681129662687542,1.5984366944433792,128.81049387486917
    133,-0.35531698596988237,2.492360542295635,561.6847180077077
    134,1.6353795648832175,1.284222908776048,193.68138178075134
    135,1.9898441309969779,2.6865478247555323,163.01533856330664
    136,-1.6765031557165804,0.32332260528005063,625.8498090545208
    137,-0.43297142094291274,-0.9245272895236081,125.7059257914247
    138,0.06464753130665857,-0.034205758861981295,1.0222255404397016
    139,1.8682929419477912,0.23849239919972565,1058.3213196742774
    140,1.9926448599061293,1.8339423072940062,457.5302852316951
    141,-1.4691937556629227,2.690585038656038,34.405143181092896
    142,-0.7247505355748336,0.05124812049083394,25.443807130351935
    143,-1.9987052105835361,1.3386643210164255,714.5098701125884
    144,-1.3494151021354481,-0.9524010187196641,774.6513186526458
    145,-1.5189084742507744,1.5733577361268831,60.180169311334026
    146,1.4048411273962365,-0.6849413019602726,706.9366996474531
    147,-1.719343440101885,1.7561854859065802,151.3843599230563
    148,-0.010869703700098654,0.4257485462253001,19.137980937916637
    149,-0.20669143604386786,1.1639375343970397,127.1686774968837
    150,-1.988058894230977,0.7523102828625632,1032.9719422199498
    151,-1.4369326388106747,2.239485822683578,8.991013569239984
    152,1.8481652649354832,-0.2991571167597429,1380.7467546684074
    153,-1.5594542600845935,1.2785513706680978,139.57155611179036
    154,0.21599302715392366,-0.9902143690519245,108.12405849965
    155,1.956954317444104,0.8232938936717837,904.7456114294331
    156,1.3329802621062652,0.862054271535893,83.79350629876686
    157,1.1783642894518689,-0.9231179644928473,534.409177274824
    158,1.6703280790473225,0.48412070657999395,532.1553766475299
    159,0.39269027675944157,0.073634684532903,1.0179932033503165
    160,0.0658446146078755,2.772334241893356,767.0543424485254
    161,0.5824134522445887,-0.16321952776248771,25.417462278363423
    162,1.0344272967939023,2.3592618150339777,166.2105173017218
    163,-1.7172129877031614,-0.624376088308507,1284.1565930828256
    164,-0.9847676481614771,2.94152415976802,392.7218057930795
    165,-1.098704140301955,1.505511560381784,13.306474123119227
    166,0.5621173297740634,0.6467930116924561,11.1357378724415
    167,0.23174627091249222,-0.49264844387562334,30.44056813195245
    168,-1.2527164203242172,-0.6684725069178432,505.8366077698906
    169,-1.9951957029171243,1.6666815032888054,544.4883663761424
    170,-0.6943972106470668,1.6015985263448815,128.17908959747615
    171,1.3704913091932305,0.3305390191131643,239.67708634023893
    172,0.2438146289447234,1.2633727335919054,145.5158770472722
    173,0.8503311613273858,0.25878022558630454,21.578258015927016
    174,-1.7303914391624367,2.166632670564697,75.95083208389319
    175,-0.7135286057018795,1.2232364137264495,53.931966886434296
    176,-0.9791140561166216,2.207147174187001,159.78783245024127
    177,0.5539071746084572,0.37202340002633427,0.6242363743002302
    178,1.9275158676689568,1.301622183019005,583.4527554430621
    179,1.0444457737442274,2.9892219251497187,360.37712737133734
    180,0.07492665350705785,-0.2701679194870321,8.461327595836613
    181,0.9778232857504707,-0.46319801623817547,201.45207185179726
    182,-0.7510049106858827,0.49125425078859486,3.595334468944874
    183,0.2546826615501834,0.9323859200380267,75.81505483358943
    184,0.7392971057713154,-0.5119649943552753,112.11552695201024
    185,0.6122714970560894,1.3688749376142586,98.95364543178735
    186,-1.1940195368434505,0.818494458514706,41.68147224590436
    187,1.350172727612108,-0.18471157071140443,403.1997020935389
    188,1.6607336712088019,-0.9950096265095985,1408.971961664554
    189,1.4576026055166968,2.677347650870205,30.761804641631173
    190,-0.6818364160093053,2.9437836030818842,617.3145201882452
    191,-1.4915450405328428,0.4607622134314511,317.3577793801536
    192,-0.9722589352035151,1.1466494769214473,7.944472416277202
    193,1.7774372162981882,1.4729046052559083,284.9916371733927
    194,1.7711912992660381,2.7644352259316967,14.484027136820007
    195,1.7026006342610547,2.2672365654724187,40.38706426343692
    196,-1.6404671152792671,-0.16135662320479538,820.6414044137362
    197,1.9907160857027923,2.0837842130886375,354.1081244805641
    198,-1.0934372341046572,0.15925326000237083,111.78496922844573
    199,-0.9608174755721932,-0.2681771618896427,145.7756639312592
    200,0.19299957138803947,1.4906339416678054,211.8840766492289
    201,0.11722509421586658,0.24183976627921222,5.982163282113635
    202,-0.7568476322237059,-0.6525332645757933,153.23516869502916
    203,1.427887339245153,2.458917475310494,17.827726505677155
    204,-0.6751621850835203,0.8262672130336934,16.52750578743188
    205,-0.26705050480302583,1.3837039293657907,173.8416320152673
    206,-1.918961302874286,2.438512436843696,163.24906730399405
    207,0.16751756811045837,2.3764369682353195,552.1794624428578
    208,-0.7377070480396184,-0.23070240121994434,63.06881046468747
    209,-1.3606030883983635,2.9580306742059546,128.07083743909394
    210,1.1530942011228755,1.1382144195686261,3.6872862067622103
    211,1.2145230943817493,-0.5929366725997705,427.7096689771458
    212,0.6365868507526793,2.092951421882615,284.96810209292863
    213,-0.5876111292107575,1.4046683254088164,114.74942243565408
    214,-0.43773592908412784,0.07835008735638294,3.3499275319678308
    215,-1.8084472983274016,1.2436723254626174,418.6829722589224
    216,0.05234289626993238,1.8933019580383248,358.32058934669595
    217,1.5622018327384293,0.9308140895820665,228.22354637037708
    218,-1.4400452341655967,2.4459249300893458,19.806706765062973
    219,1.477789992437589,0.07225226095862203,446.11838514263655
    220,-1.175819478520984,0.5100741395635642,80.85585543987189
    221,-1.7868844552685341,2.9874316826565295,11.990751790583754
    222,-0.1701585948685289,2.111039619446777,434.8773457082143
    223,1.0056044814300216,2.050864314476477,108.08182536432041
    224,-1.3427309475356597,-0.32844169152687286,459.761381371964
    225,-0.34834705333290295,1.822305224282304,291.1443804543967
    226,-1.7040160391288084,2.7721604654332426,9.041195909664287
    227,0.45086851526963034,0.825194428672809,38.97900028178963
    228,-0.05219477503475911,0.9312632671680396,87.32557621335144
    229,-1.716202146817106,0.929518198628791,413.7354621345851
    230,0.5264041025981201,2.982887629756693,732.352270542538
    231,1.7774140535261411,0.01344075933578992,990.1849441579069
    232,0.666763637714856,2.3325537883885477,356.5579095451704
    233,-1.7559092220221042,-0.4253422586616442,1238.5939803171473
    234,0.714801593374891,1.0621048937994408,30.459466871928257
    235,-0.14859246272148985,2.502779724905236,616.7065160901996
    236,1.9300226406305843,-0.699426093651581,1958.4084125052734
    237,-0.2859493053691353,0.7269440653274533,43.2790095021475
    238,0.9660423756593763,1.741511122704626,65.33171797028749
    239,-0.4443952827752926,2.31532330047832,450.609266410053
    240,0.5508438543149738,2.582365962369495,519.5571310408783
    241,0.5660708856554741,-0.36064637577977665,46.57564846146193
    242,-0.2797011479748681,2.865115769584837,778.3093414464681
    243,-1.3325286322696148,2.0305539724206145,11.939182684609094
    244,1.3796208232785165,-0.390650573213875,526.3896339934172
    245,-0.7004970935404269,1.0286708211018603,31.833362020893322
    246,1.019687048270392,1.297118325473289,6.623632061558298
    247,1.0621251221218881,0.16225729278120937,93.29096127981595
    248,-1.0131516229351987,2.7485971971759793,300.62284854033345
    249,0.8667865637816026,0.6987871693499295,0.29370458749706696
    250,-1.0935761492205978,1.3119894088131394,5.730532002778362
    251,0.026038785782890983,1.2739866543012637,163.0800886817741
    252,-0.8055084498911587,0.6765877842770704,3.336833280265398
    253,0.3698835227087307,-0.7713414852976004,82.87165269643317
    254,1.3178251272928172,1.5588082203615383,3.264247427669332
    255,0.018525148823502313,-0.6307004151847324,40.784894929602444
    256,-0.15092964060440472,1.7154157150161948,287.826287873286
    257,1.2412412658416287,2.727376769600169,140.88314812034034
    258,-1.1910190805822403,2.1894275805049843,64.22941986491709
    259,-1.7498564635901377,0.5184424309124536,654.5290223496775
    260,-0.8189423618588285,2.320840090487571,275.61580881135137
    261,-1.3827205555906743,0.8850400948571133,111.12479739779423
    262,-0.48138382386601464,0.43476450372991415,6.316783334726612
    263,0.7371233655130518,2.798374614947416,508.58231947332774
    264,1.7582150325064143,2.972848279853419,1.978447264997389
    265,0.8116049543054178,1.4002419998421907,55.02356057024467
    266,0.37118962068620487,-0.36689093840918563,25.864853171784638
    267,-0.5021284933288688,1.0490330089501545,65.76134864167483
    268,1.6606069568154123,-0.30634747174212595,939.2232893388866
    269,1.1717256785214447,1.9478389321743466,33.080245395587255
    270,1.5943910938560708,2.0866425766415335,21.09589506708865
    271,-0.489175626705173,0.8448891914399486,38.89234373517525
    272,-1.4756686587030248,1.7754693976482736,22.299675808751775
    273,0.1242851427581817,2.557821771985136,647.1339280015045
    274,-0.058453045369466494,-0.8099418438104906,67.2755444466084
    275,1.203473590180987,0.5075416540061095,88.55318794367878
    276,0.9520115250269332,2.5446927929200864,268.4268961285059
    277,0.9817374320219301,-0.2558456166462024,148.75592200017402
    278,1.454247143197633,1.358917605850447,57.347413877936496
    279,-1.4704005950411512,-0.09332351923390991,514.7864397527222
    280,-0.9616370755948012,1.9600350027106006,111.03037984968093
    281,-1.3357359468514556,1.1699922296362617,43.17961636650628
    282,1.7536957521670309,-0.46254641832770194,1252.309067528775
    283,-0.2363355062973791,-0.9578263555887481,104.28340741233293
    284,-0.5662410273876688,2.130267861917245,329.93242780771453
    285,-0.35662605177003215,-0.2028782258135884,12.734418805322186
    286,1.3505451088985696,2.0399414469519437,4.787158137066003
    287,-1.375684913193663,-0.5387251731568794,596.7338297434035
    288,0.3370706426604695,0.5171521108075585,16.723564716781663
    289,0.6813338441404717,-0.9411976599694807,197.62024948093838
    290,-0.3271888024752698,-0.5576196708122039,45.940341239690966
    291,-1.478135331578379,0.2836491462298021,367.61057387003723
    292,-0.5667320620798111,-0.3165302453841812,43.12275213293167
    293,0.3545323117364667,2.798207617020191,714.6499807999336
    294,-1.8119872724638921,1.9108276518146243,196.27472393032363
    295,0.21668252034187452,0.0834871512764348,0.7470730099440539
    296,-0.3718990324975322,0.28363704396673084,3.994134178075578
    297,0.6738759740941052,1.7739288968969613,174.29885818482998
    298,1.5703787722925715,-0.8384676490990457,1092.335119487834
    299,-0.7163547336312304,1.873516575319285,188.00175808640213"""

    def setUp(self):
        if not PANDAS_PLUS:
            self.skipTest("Pandas not installed")
        self.tmp_dir = tempfile.mkdtemp()
        self.definitions = self.RESAMPLE_SAMPLER.format(path=self.tmp_dir)
        self.csv_data = self.CSV1
        self.sampler_file = os.path.join(self.tmp_dir, "config.yaml")
        self.csv_file = os.path.join(self.tmp_dir, "samples.csv")
        with open(self.sampler_file, 'w') as _file:
            _file.write(self.definitions)
        with open(self.csv_file, 'w') as _file:
            _file.write(self.csv_data)

        self.sample_data = read_yaml(self.sampler_file)

        self.sampler = new_sampler(self.sample_data['sampler'])

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_setup(self):
        self.assertTrue(os.path.isdir(self.tmp_dir))
        self.assertTrue(os.path.isfile(self.sampler_file))
        self.assertTrue(os.path.isfile(self.csv_file))

    def test_dispatch(self):
        self.assertTrue(isinstance(self.sampler, BestCandidateSampler))

    def test_samples(self):
        # sampler = new_sampler_from_yaml(yaml_text)
        # self.assertTrue(isinstance(sampler, BestCandidateSampler))
        samples = self.sampler.get_samples()
        # self.assertEqual(len(samples), 5)
        # for sample in samples:
        #     self.assertEqual(sample['X1'], 20)
        #     self.assertEqual(sample['X2'], "foo")
        #     self.assertTrue(sample['X3'] > 5)
        #     self.assertTrue(sample['X4'] > 5)
        #     self.assertTrue(sample['X3'] < 10)
        #     self.assertTrue(sample['X4'] < 10)


class TestCsvRowSampler(unittest.TestCase):
    """Unit test for testing the csv sampler."""
    CSV_SAMPLER = """
    sampler:
        type: csv
        csv_file: {path}/test.csv
        row_headers: True
    """

    # Note: the csv reader does not ignore blank lines
    CSV1 = """X1,20,20
    X2,5,10
    X3,5,10"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.definitions = self.CSV_SAMPLER.format(path=self.tmp_dir)
        self.csv_data = self.CSV1
        self.sampler_file = os.path.join(self.tmp_dir, "config.yaml")
        self.csv_file = os.path.join(self.tmp_dir, "test.csv")
        with open(self.sampler_file, 'w') as _file:
            _file.write(self.definitions)
        with open(self.csv_file, 'w') as _file:
            _file.write(self.csv_data)

        self.sample_data = read_yaml(self.sampler_file)

        self.sampler = new_sampler(self.sample_data['sampler'])

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_setup(self):
        self.assertTrue(os.path.isdir(self.tmp_dir))
        self.assertTrue(os.path.isfile(self.sampler_file))
        self.assertTrue(os.path.isfile(self.csv_file))

    def test_dispatch(self):
        self.assertTrue(isinstance(self.sampler, CsvSampler))

    def test_samples(self):
        samples = self.sampler.get_samples()
        self.assertEqual(len(samples), 2)
        for sample in samples:
            self.assertEqual(sample['X1'], 20)
        self.assertEqual(samples[0]['X2'], 5)
        self.assertEqual(samples[0]['X3'], 5)
        self.assertEqual(samples[1]['X2'], 10)
        self.assertEqual(samples[1]['X3'], 10)

class TestCsvColumnSampler(unittest.TestCase):
    """Unit test for testing the csv sampler."""
    CSV_SAMPLER = """
    sampler:
        type: csv
        csv_file: {path}/test.csv
        row_headers: False
    """

    # Note: the csv reader does not ignore blank lines
    CSV1 = """X1,X2,X3
    20,5,5
    20,10,10"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.definitions = self.CSV_SAMPLER.format(path=self.tmp_dir)
        self.csv_data = self.CSV1
        self.sampler_file = os.path.join(self.tmp_dir, "config.yaml")
        self.csv_file = os.path.join(self.tmp_dir, "test.csv")
        with open(self.sampler_file, 'w') as _file:
            _file.write(self.definitions)
        with open(self.csv_file, 'w') as _file:
            _file.write(self.csv_data)

        self.sample_data = read_yaml(self.sampler_file)

        self.sampler = new_sampler(self.sample_data['sampler'])

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_setup(self):
        self.assertTrue(os.path.isdir(self.tmp_dir))
        self.assertTrue(os.path.isfile(self.sampler_file))
        self.assertTrue(os.path.isfile(self.csv_file))

    def test_dispatch(self):
        self.assertTrue(isinstance(self.sampler, CsvSampler))

    def test_samples(self):
        samples = self.sampler.get_samples()
        self.assertEqual(len(samples), 2)
        for sample in samples:
            self.assertEqual(sample['X1'], 20)
        self.assertEqual(samples[0]['X2'], 5)
        self.assertEqual(samples[0]['X3'], 5)
        self.assertEqual(samples[1]['X2'], 10)
        self.assertEqual(samples[1]['X3'], 10)

class TestCustomSampler(unittest.TestCase):
    """Unit test for testing the custom sampler."""

    CUSTOM_SAMPLER = """
        sampler:
            type: custom
            function: test_function
            module: {path}/codepy_sampler_test.py
            args:
                num_samples: 2
    """

    CUSTOM_FUNCTION = (
        """def test_function(num_samples):
               return [{"X1": 20, "X2": 5, "X3": 5},
                       {"X1": 20, "X2": 10, "X3": 10}][:num_samples]
        """)

    def setUp(self):
        print("CUSTOM_FUNCTION:\n" + self.CUSTOM_FUNCTION)
        self.tmp_dir = tempfile.mkdtemp()
        self.definitions = self.CUSTOM_SAMPLER.format(path=self.tmp_dir)
        self.function_data = self.CUSTOM_FUNCTION
        self.sampler_file = os.path.join(self.tmp_dir, "config.yaml")
        self.function_file = os.path.join(self.tmp_dir,
                                          "codepy_sampler_test.py")
        with open(self.sampler_file, 'w') as _file:
            _file.write(self.definitions)
        with open(self.function_file, 'w') as _file:
            _file.write(self.function_data)

        self.sample_data = read_yaml(self.sampler_file)

        self.sampler = new_sampler(self.sample_data['sampler'])

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_setup(self):
        self.assertTrue(os.path.isdir(self.tmp_dir))
        self.assertTrue(os.path.isfile(self.sampler_file))
        self.assertTrue(os.path.isfile(self.function_file))

    def test_dispatch(self):
        self.assertTrue(isinstance(self.sampler, CustomSampler))

    def test_samples(self):
        samples = self.sampler.get_samples()
        self.assertEqual(len(samples), 2)

        for sample in samples:
            self.assertEqual(sample['X1'], 20)
        self.assertEqual(samples[0]['X2'], 5)
        self.assertEqual(samples[0]['X3'], 5)
        self.assertEqual(samples[1]['X2'], 10)
        self.assertEqual(samples[1]['X3'], 10)
