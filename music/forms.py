from django import forms
from .models import MusicInfo


class MusicForm(forms.ModelForm):
  class Meta:
    model = MusicInfo
    fields = {'songname', 'artist'}

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
