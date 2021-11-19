from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [

    path('', views.home, name='homepage' ),
    path( 'fertilizer', views.fertilizer, name='fert_pred'),
    path( 'fert_result', views.fertpred, name='fert_res'),
    # path('register', views.register, name='register')
    # path('register', views.register, name='register')
    # path('register', include('accounts.urls')),
    path( 'about', views.about, name='aboutus'),
    path( 'contact', views.contact, name='contactus'),

]