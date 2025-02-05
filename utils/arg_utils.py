import argparse

def add_bool_arg(parser, name, default=False, **kwargs):
    r"""
    Args:
        :param parser --> an instance of argparser.ArgumentParser (can also be parser.add_argument_group)
        :param name --> the name of the boolean argument to be added
        :param default --> default value of the boolean argument
        :param **kwargs --> allows additional keyword arguments (e.g., "help" desciptions) to be passed 
    """
    # create a mutually exclusive group
    # only one of the arguments inside this group can be provided at a time
    # providing neither argument is acceptable (use default value)
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--" + name,
        dest=name,
        action='store_true',
        help="Default: " + ("Enabled" if default else "Disabled")
    )
    group.add_argument("--no--" + name, dest=name, action="store_false", **kwargs)
    parser.set_defaults(**{name: default})