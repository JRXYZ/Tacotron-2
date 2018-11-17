import os
import argparse
from time import sleep
from time import time

import tensorflow as tf

from hparams import hparams
from hparams import hparams_debug_string
from infolog import log
from tqdm import tqdm

from synthesize import prepare_run
from tacotron.synthesizer import Synthesizer

def load_synth(args, checkpoint_path, output_dir):
    eval_dir = os.path.join(output_dir, 'eval')
    log_dir = os.path.join(output_dir, 'logs-eval')

    #Create output path if it doesn't exist
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)
    log(hparams_debug_string())
    
    synth = Synthesizer()
    synth.load(checkpoint_path, hparams)
    return synth, eval_dir, log_dir

def prepare_network(args, taco_checkpoint):
    output_dir = 'tacotron_' + args.output_dir
    try:
        checkpoint_path = tf.train.get_checkpoint_state(taco_checkpoint).model_checkpoint_path
        log('loaded model at {}'.format(checkpoint_path))
    except:
        raise RuntimeError('Failed to load checkpoint at {}'.format(taco_checkpoint))

    synth, eval_dir, log_dir = load_synth(args, checkpoint_path, output_dir)
    
    return synth, output_dir, eval_dir, log_dir

def synthesize_sentence(sentences, synth, eval_dir, log_dir):
    log('Starting Synthesis')
    with open(os.path.join(eval_dir, 'map.txt'), 'w') as file:
        for i, texts in enumerate(tqdm(sentences)):
            start = time()
            basenames = ['batch_{}_sentence_{}'.format(i, j) for j in range(len(texts))]
            mel_filenames, speaker_ids = synth.synthesize(texts, basenames, eval_dir, log_dir, None)

            for elems in zip(texts, mel_filenames, speaker_ids):
                file.write('|'.join([str(x) for x in elems]) + '\n')
    log('synthesized mel spectrograms at {}'.format(eval_dir))
    return eval_dir


def main():
    accepted_modes = ['eval', 'synthesis', 'live']
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='pretrained/', help='Path to model checkpoint')
    parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--name', default='', help='Name of logging directory if the two models were trained together.')
    parser.add_argument('--tacotron_name', default='', help='Name of logging directory of Tacotron. If trained separately')
    parser.add_argument('--wavenet_name', default='', help='Name of logging directory of WaveNet. If trained separately')
    parser.add_argument('--model', default='Tacotron')
    parser.add_argument('--input_dir', default='training_data/', help='folder to contain inputs sentences/targets')
    parser.add_argument('--mels_dir', default='tacotron_output/eval/', help='folder to contain mels to synthesize audio from using the Wavenet')
    parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')
    parser.add_argument('--mode', default='eval', help='mode of run: can be one of {}'.format(accepted_modes))
    parser.add_argument('--GTA', default='True', help='Ground truth aligned synthesis, defaults to True, only considered in synthesis mode')
    parser.add_argument('--text', default='', help='Text file contains list of texts to be synthesized. Valid if mode=eval')
    args = parser.parse_args()

    taco_checkpoint, wave_checkpoint, hparams = prepare_run(args)
    sentences = []
    synth, output_dir, eval_dir, log_dir = prepare_network(args, taco_checkpoint)

    sentences = [[args.text]]

    tic = time()
    synthesize_sentence(sentences, synth, eval_dir, log_dir)
    toc = time()

    print('time taken to synthesize audio: {} secs'.format(toc-tic))

if __name__ == '__main__':
    main()