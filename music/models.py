from django.db import models

# Create your models here.
class MusicInfo(models.Model):
  songname = models.TextField(max_length=100)
  artist = models.TextField(max_length=100)
  feeling = models.TextField(max_length=100)
  date = models.DateTimeField(auto_now_add=True)
  fs = models.JSONField()

  def __str__(self):
    return str(self.songname) + ' by ' + str(self.artist) + ':' + str(self.feeling)

  def get_fs(self):
    return self.fs

  class Meta:
    ordering = ('-date',)