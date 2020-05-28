from argparse import ArgumentParser


def add_boolean_argument(parser: ArgumentParser, name, description, default):
    group = parser.add_mutually_exclusive_group(required=False)
    kebab_name = name.replace('_', '-')
    group.add_argument('--' + kebab_name, dest=name, action='store_true',
                       help=f'enable {description}')
    group.add_argument('--no-' + kebab_name, dest=name, action='store_false',
                       help=f'disable {description}')
    parser.set_defaults(**{name: default})
