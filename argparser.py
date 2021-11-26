import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Caliope-Toolkit: Punctuation restoration')
    parser.add_argument("-i", "--input", default='data/test.eaf', action='store', dest='input', help="Input file")
    parser.add_argument("-o", "--output", default='', action='store', dest='output', help="Output file")
    parser.add_argument('--name', default='punctuation-restore', type=str, help='name of run')
    parser.add_argument('--language', default='gl', type=str,
                        help='language, available options are Galician, Spanish, and English')

    parser.add_argument('--pretrained-model', default='bertinho-gl-base-cased', type=str, help="pretrained BERT's LM")
    parser.add_argument('--lstm-dim', default=-1, type=int,
                        help='hidden dimension in LSTM layer, if -1 is set equal to hidden dimension in language model')
    parser.add_argument('--use-crf', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to use CRF layer or not')
    parser.add_argument('--data-path', default='../data/gl/test', type=str, help='path to test datasets')
    parser.add_argument('--weight-path', default='out/weights.pt', type=str, help='model weight path')
    parser.add_argument('--sequence_length', default=96, type=int,
                        help='sequence length to use when preparing dataset (default 256)')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size (default: 8)')
    parser.add_argument('--save-path', default='out/', type=str, help='model and log save directory')

    args = parser.parse_args()
    return args
