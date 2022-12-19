import time
starttime = time.time()

import sys
out_path = sys.argv[1]
id = int(sys.argv[2])
text = ' '.join(sys.argv[3:])

print('发言人：', id, '\n发言内容：', text, '\n正在加载模型……', sep='')

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

stn_tst_mt = get_text(text, hps_mt)

with torch.no_grad():
  x_tst_mt = stn_tst_mt.unsqueeze(0)
  x_tst_mt_lengths = torch.LongTensor([stn_tst_mt.size(0)])
  sid_mt = torch.LongTensor([id])
  audio_mt = net_g_mt.infer(x_tst_mt, x_tst_mt_lengths, sid=sid_mt, noise_scale=.667, noise_scale_w=.8, length_scale=1.2)[0][0,0].data.cpu().float().numpy()

print('正在输出到文件：', out_path, sep='')
write(out_path, hps_mt.data.sampling_rate, audio_mt)

print('生成用时：', time.time() - starttime, 's', sep='')