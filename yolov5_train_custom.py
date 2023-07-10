import torch
import utils
import argparse
import subprocess
import comet_ml
import os

script_path = os.path.realpath(__file__)
script_directory = os.path.dirname(script_path)

# argparser with default values
def argparser():
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--img', type=int, default=500, help='image size')
  parser.add_argument('-b', '--batch', type=int, default=16, help='batch size')
  parser.add_argument('-e', '--epochs', type=int, default=500, help='epochs')
  parser.add_argument('-d', '--data', type=str, default='triplets.yaml', help='dataset.yaml path')
  parser.add_argument('-w', '--weights', type=str, default='yolov5s.pt', help='weights path')
  parser.add_argument('--cache',  action='store_true', help='cache images for faster training')
  parser.add_argument('--comet', action='store_true', help='log training with Comet')
  parser.add_argument('-n', '--name', type=str, default='exp', help='experiment name')
  return parser.parse_args()

if __name__ == '__main__':
  args = argparser()
  #@ title Select YOLOv5 🚀 logger {run: 'auto'}
  if args.comet:
    logger = 'Comet' #@param ['Comet', 'ClearML', 'TensorBoard']

    if logger == 'Comet':
      command = 'pip install -q comet_ml'
      subprocess.run(command, shell=True)
      comet_ml.init()

  if args.cache:
    command=f"python {script_directory}/train.py --img {args.img} --batch {args.batch} --epochs {args.epochs} --data {args.data} --weights {args.weights} --cache --name {args.name}"
  else:
    command=f"python {script_directory}/train.py --img {args.img} --batch {args.batch} --epochs {args.epochs} --data {args.data} --weights {args.weights} --name {args.name}" 
  subprocess.run(command, shell=True)