import argparse

import pytest

from aspset510.util import add_boolean_argument


class TestAddBooleanArgument():
    @pytest.fixture
    def parser(self):
        parser = argparse.ArgumentParser()
        add_boolean_argument(parser, 'skip_checksum', default=False,
                             description='skip checking file integrity')
        return parser

    def test_smoke(self, parser):
        assert isinstance(parser, argparse.ArgumentParser)

    def test_default(self, parser):
        assert parser.get_default('skip_checksum') == False

    def test_option_names(self, parser):
        all_option_strings = set()
        for action in parser._optionals._group_actions:
            all_option_strings.update(action.option_strings)
        assert '--skip-checksum' in all_option_strings
        assert '--no-skip-checksum' in all_option_strings

    def test_enable_option(self, parser):
        opts = parser.parse_args(['--skip-checksum'])
        assert opts.skip_checksum == True

    def test_disable_option(self, parser):
        opts = parser.parse_args(['--no-skip-checksum'])
        assert opts.skip_checksum == False

    def test_mutual_exclusion(self, parser):
        namespace = argparse.Namespace()
        with pytest.raises(argparse.ArgumentError):
            parser._parse_known_args(['--skip-checksum', '--no-skip-checksum'], namespace)
