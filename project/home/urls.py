from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [

    path('', views.home, name='homepage' ),
    path( 'pest_prediction', views.pest_prediction, name='pest_pred'),
    path( 'pest_result', views.pest_pre_result, name='pest_res'),
    path( 'fertilizer', views.fertilizer, name='fert_pred'),
    path( 'fert_result', views.fertpred, name='fert_res'),
    # path('register', views.register, name='register')
    # path('register', views.register, name='register')
    # path('register', include('accounts.urls')),
    path( 'about', views.about, name='aboutus'),
    path( 'contact', views.contact, name='contactus'),
    path( 'dis_pred', views.disease_pred, name="dis_pred"),
    path('profile', views.profile, name='my_profile'),
    path( 'editprofile', views.edit_profile, name = 'edit_profile'),
    path('chatbot', views.chatbot, name='chatbot'),
    path('chatbot_index', views.chatbot_index, name='chatbot_index'),
]
