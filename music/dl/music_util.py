import numpy as np
import torch
import librosa

def min_max(x, axis=None): # do min_max normalization ?->https://atmarkit.itmedia.co.jp/ait/articles/2110/07/news027.html
  min = 0
  max = 159.42822
  # max = 137.74599
  x = np.abs(x)
  result = (x-min)/(max-min)
  return result

def quality_prediction(fileobj, scaler, model): # return prediction value
  if torch.cuda.is_available():
    devi = 'cpu'
  else:
    devi = 'cpu'
  data, sr = librosa.load(fileobj) 

  sum = torch.zeros(4, device=devi)
  cnt = 0
  for i in range(46):
    start = sr//2*i
    stop  = 5*sr + sr//2*i

    if stop >= len(data):
      break

    # STFT
    n_fft=512
    hop_length=256
    
    y = data[start:stop]

    x = []
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    # Normalization
    stft = min_max(stft)

    # Data Transform
    stft = np.expand_dims(stft, axis=2)
    stft = np.expand_dims(stft, axis=0) # (Batch, Height, Width, Color)
    stft = torch.from_numpy(stft.astype(np.float32)).clone()
    stft = stft.permute(0,3,2,1)
    stft = stft.to(devi)

    
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y) 
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    r = [np.mean(chroma_stft), np.mean(rmse), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)]
    for e in mfcc:
      r.append(np.mean(e))

    r = np.array(r, dtype=float)
    r = np.expand_dims(r,0)
    r = scaler.transform(r)
    r = torch.from_numpy(r).float()
    r = r.to(devi)
    
    x.append(r)
    x.append(stft)


    # Predict
    scores = model(x)


    # Get sum
    sum = sum + scores
    cnt+=1
  return sum/cnt