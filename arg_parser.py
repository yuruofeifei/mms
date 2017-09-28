import argparse
from export_model import export_model

class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, 'models', {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in values})

class ArgParser(object):
    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(prog='mxnet-model-serving', description='MXNet Model Serving')

        parser.add_argument('--models',
                            required=True,
                            action=StoreDictKeyPair,
                            metavar='KEY1=VAL1,KEY2=VAL2...',
                            nargs="+",
                            help='Models to be deployed')

        parser.add_argument('--process', help='Using user defined model service')

        parser.add_argument('--gen-api', help='Generate API')

        parser.add_argument('--port', help='Port')

        subparsers = parser.add_subparsers(help='sub-command help')

        parser_export = subparsers.add_parser('export', help='export help')

        parser_export.add_argument('--models',
                                   required=True,
                                   metavar='KEY=VAL',
                                   help='Model to be exported')

        parser_export.add_argument('--signature',
                                   required=True,
                                   type=str,
                                   help='Path to signature file')

        parser_export.add_argument('--synset',
                                   type=str,
                                   help='Path to synset file')

        parser_export.set_defaults(func=export_model)
        return parser.parse_args()


