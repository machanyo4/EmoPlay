from django.contrib import admin
from django.urls import path, include # 2)ディレクトリ構造helloを認識させるために、includeをimportする。
# import hello.views as hello # 1)まずappをimportする

urlpatterns = [
    path('admin/', admin.site.urls),
    path('hello/', include('hello.urls')), # includeは多分アプリケーションの階層を理解してる。したがってhello.urlsが使用可 
    path('sns/', include('sns.urls')),
    path('music/', include('music.urls'))
]
