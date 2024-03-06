from Form_Meaning_Pairing import FMP
import argparse, numpy

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('lang', type=str, help='Required argument: Language in question.')
    parser.add_argument('--min', type=int, help='Minimum frequency of a form. (default=100)', default=100)
    return parser

if __name__ == "__main__":
    parser = get_parser().parse_args()
    lang = parser.lang
    min_freq = parser.min
    fmp = FMP(lang, min_freq)

    fmp.tokenize()
    token_indice = fmp.contextualize()
    embeddings = fmp.embed(token_indice)
    print(f'-----{lang}-----')
    for e in numpy.arange(0.5,15.0,0.5):
        estimations = fmp.estimate(embeddings, epsilon=e, min_samples=2)
        entropy = fmp.entropy(estimations)
        print(f'epsilon={e}: entropy={entropy}')