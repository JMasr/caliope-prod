import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Caliope-Toolkit: Punctuation restoration')
    parser.add_argument("-i", "--input", default='data/test.eaf', action='store', dest='input', help="Input file")
    parser.add_argument("-o", "--output", default='', action='store', dest='output', help="Output file")
    parser.add_argument('--name', default='punctuation-restore', type=str, help='name of run')
    parser.add_argument('--language', default='galician', type=str,
                        help='language, available options are Galician, Spanish, and English')

    args = parser.parse_args()
    return args
