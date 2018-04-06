import argparse
import os
import re
from tacotron.hparams import hparams, hparams_debug_string
from tacotron.synthesizer import Synthesizer
import tensorflow as tf 
import time
from tqdm import tqdm


def run_eval(args, checkpoint_path):
	print(hparams_debug_string())
	synth = Synthesizer()
	synth.load(checkpoint_path)
	eval_dir = os.path.join(args.output_dir, 'eval')
	log_dir = os.path.join(args.output_dir, 'logs-eval')

	#Create output path if it doesn't exist
	os.makedirs(eval_dir, exist_ok=True)
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
	os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)

	with open(os.path.join(eval_dir, 'map.txt'), 'w') as file:
		file.write('"input"|"generated_mel"\n')
		for i, text in enumerate(tqdm(hparams.sentences)):
			start = time.time()
			mel_filename = synth.synthesize(text, i+1, eval_dir, log_dir, None)

			file.write('"{}"|"{}"\n'.format(text, mel_filename))
	print('synthesized mel spectrograms at {}'.format(eval_dir))

def run_synthesis(args, checkpoint_path):
	metadata_filename = os.path.join(args.input_dir, 'train.txt')
	print(hparams_debug_string())
	synth = Synthesizer()
	synth.load(checkpoint_path, gta=args.GTA)
	with open(metadata_filename, encoding='utf-8') as f:
		metadata = [line.strip().split('|') for line in f]
		hours = sum([int(x[1]) for x in metadata]) * hparams.frame_shift_ms / (3600 * 1000)
		#Making sure we got all of it
		print('Loaded metadata for {} examples ({:.2f} hours)'.format(len(metadata), hours))

	if args.GTA==True:
		synth_dir = os.path.join(args.output_dir, 'gta')
	else:
		synth_dir = os.path.join(args.output_dir, 'natural')

	#Create output path if it doesn't exist
	os.makedirs(synth_dir, exist_ok=True)

	print('starting synthesis')
	with open(os.path.join(synth_dir, 'map.txt'), 'w') as file:
		file.write('"input"|"frames"|"target_mel"|"generated_mel"\n')
		for i, meta in enumerate(tqdm(metadata)):
			text = meta[2]
			mel_filename = os.path.join(args.input_dir, meta[0])
			mel_output_filename = synth.synthesize(text, i+1, synth_dir, None, mel_filename)

			file.write('"{}"|"{}"|"{}"|"{}"\n'.format(text, meta[1], mel_filename, mel_output_filename))
	print('synthesized mel spectrograms at {}'.format(synth_dir))

def tacotron_synthesize(args):
	hparams.parse(args.hparams)
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

	try:
		checkpoint_path = tf.train.get_checkpoint_state(args.checkpoint).model_checkpoint_path
		print('loaded model at {}'.format(checkpoint_path))
	except:
		raise AssertionError('Cannot restore checkpoint: {}, did you train a model?'.format(args.checkpoint))

	if args.mode == 'eval':
		run_eval(args, checkpoint_path)
	else:
		run_synthesis(args, checkpoint_path)
