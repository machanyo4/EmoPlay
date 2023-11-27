from django.shortcuts import render, redirect
from django.http import JsonResponse, Http404
from .dl.model_def import MultiModalNet
from .models import MusicInfo
from .forms import MusicForm
from .dl.music_util import quality_prediction
import pickle
import torch
import traceback
import json
import os
import shutil
from django.http import HttpResponse
from django.utils.http import quote


# Create your views here.
if torch.cuda.is_available():
  devi = 'cpu'
else:
  devi = 'cpu'

# Process
num_1d_features = 26
model     = MultiModalNet(input_ch=1, num_classes=4, num_1d_features=num_1d_features).to(devi)
model_path = './music/dl/mul_model.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device(devi)))
scaler = pickle.load(open("./music/dl/1d_scaler_5s_0.5shift.pickle", "rb"))
model = model.eval()

def index(request):
  data = MusicInfo.objects.all()
  params = {
    'data': data
  }
  return render(request, 'music/index.html', params)

def ajax(request):
  try:
    file = request.FILES['uploadfile']
    songname, extension = os.path.splitext(file.name)
    #audio = io.BytesIO(file.read())
    with open('./music/dl/tmp/' + file.name, 'wb') as f:
      f.write(file.read())
    audio = './music/dl/tmp/' + file.name
    score = quality_prediction(audio, scaler, model)
    print(score)
    fs = ['Happy', 'Tense', 'Melancholy', 'Relaxed']
    f = fs[torch.argmax(score)]
    d = {
        'Happy': round(100*score[0,0].item()),
        'Tense': round(100*score[0,1].item()),
        'Melancholy': round(100*score[0,2].item()),
        'Relaxed': round(100*score[0,3].item()),
    }
    obj = MusicInfo(songname=songname, artist='unknown', feeling=f, fs=json.dumps(d))
    obj.save()
    p = {
      'id': obj.id
    }
    destination_folder = os.path.join('./music/emotions/', f)
    os.makedirs(destination_folder, exist_ok=True)
    filepath = os.path.join(destination_folder, file.name)
    shutil.move(audio, filepath)
    return JsonResponse(p)
  except Exception:
    print(traceback.format_exc())
    return Http404("Error")

def emotion_folder(request, emotion):
    folder_path = os.path.join('./music/emotions/', emotion)
    
    if os.path.exists(folder_path):
        files_with_info = []
        for file_name in os.listdir(folder_path):
            file_name = file_name
            base_name, extension = os.path.splitext(file_name)
            extension = extension.lstrip('.')
            encoded_file_name = quote(file_name)
            files_with_info.append({'file_name': base_name, 'extension': extension})

        params = {
            'emotion_folder': emotion,
            'files': files_with_info
        }
        return render(request, 'music/folder.html', params)
    else:
        return HttpResponse(f'{emotion} folder does not exist.')


def detail(request, page=0):
  data = MusicInfo.objects.get(id=page)
  params = {
    'data': data,
    'fs' : json.loads(data.get_fs())
  }
  return render(request, 'music/detail.html', params)

def edit(request, page=0):
  data = MusicInfo.objects.get(id=page)
  org_songname = data.songname
  org_filename = data.songname + '.mp3'
  org_artist = data.artist
  if request.method == 'POST':
        minfo = MusicForm(request.POST, instance=data)
        if minfo.is_valid():
            new_songname = minfo.cleaned_data['songname']
            artist = minfo.cleaned_data['artist']
            new_filename = new_songname + '-' + artist + '.mp3'
            emotion_folder = os.path.join('./music/emotions/', data.feeling)
            org_filepath = os.path.join(emotion_folder, org_filename)
            new_filepath = os.path.join(emotion_folder, new_filename)
            try:
                os.rename( org_filepath, new_filepath)
            except FileNotFoundError:
                try:
                  re_filename = org_songname + '-' + org_artist
                  os.rename(re_filename, new_filepath)
                except FileNotFoundError:
                   print(f"Error: File not found - {org_filename}")
            data.songname = new_songname
            data.artist = artist
            data.save()
            return redirect(to='/music')
  else:
      form = MusicForm(instance=data, initial={'songname': data.songname, 'artist': data.artist})
  params = {
    'data': data,
    'form' : MusicForm(instance=data, initial={'songname':data.songname, 'artist':data.artist})
  }
  return render(request, 'music/edit.html', params)

def clear_database(request):
    MusicInfo.objects.all().delete()
    return JsonResponse({'message': 'Database has been cleared.'})