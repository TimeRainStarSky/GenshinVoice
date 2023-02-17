import flask
server = flask.Flask(__name__)
import sys
import re
import cn2an

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write
import io

def get_text(text, hps):
  text_norm = text_to_sequence(text, hps.data.text_cleaners)
  if hps.data.add_blank:
    text_norm = commons.intersperse(text_norm, 0)
  text_norm = torch.LongTensor(text_norm)
  return text_norm

hps_mt = utils.get_hparams_from_file('configs/genshin.json')

net_g_mt = SynthesizerTrn(
  len(symbols),
  hps_mt.data.filter_length // 2 + 1,
  hps_mt.train.segment_size // hps_mt.data.hop_length,
  n_speakers=hps_mt.data.n_speakers,
  **hps_mt.model)
_ = net_g_mt.eval()

_ = utils.load_checkpoint('G_809000.pth', net_g_mt, None)

blacklist = []

@server.route('/')
def main():
  id   = flask.request.args.get('id', type=int, default=0)
  text = flask.request.args.get('text', type=str, default='要说的话被你吃了？',)
  bot_id = flask.request.args.get('bot_id', type=int)
  user_id = flask.request.args.get('user_id', type=int)
  print('BotID：', bot_id, '\n用户ID：', user_id, '\n发言人：', id, '\n发言内容：', text, sep='')
  if not user_id or user_id in blacklist:
    return '无效请求'
  try:
    numbers = re.findall(r'\d+(?:\.?\d+)?', text)
    for number in numbers:
      text = text.replace(number, cn2an.an2cn(number), 1)

    stn_tst_mt = get_text(text, hps_mt)

    with torch.no_grad():
      x_tst_mt = stn_tst_mt.unsqueeze(0)
      x_tst_mt_lengths = torch.LongTensor([stn_tst_mt.size(0)])
      sid_mt = torch.LongTensor([id])
      audio_mt = net_g_mt.infer(x_tst_mt, x_tst_mt_lengths, sid=sid_mt, noise_scale=.667, noise_scale_w=.8, length_scale=1.2)[0][0,0].data.cpu().float().numpy()

    bytes = io.BytesIO()
    write(bytes, hps_mt.data.sampling_rate, audio_mt)
  except Exception as e:
    print('错误：', e, sep='')
    return e
  return flask.Response(bytes, mimetype='audio/wav')

if __name__ == '__main__':
  server.run(host='0.0.0.0', port=sys.argv[1])