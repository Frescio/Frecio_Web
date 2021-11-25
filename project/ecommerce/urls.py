from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    
    path( '/mycrops', views.mycrops , name='my_crops' ),
    path( '/allcrops', views.viewcrops , name='view_all_crops' ),
    path( '/wishlist', views.mywishlist , name='my_wishlist' ),
    path( '/mandi', views.mandi, name='mandi_pred'),
    path( '/mandi_result', views.mandipred, name='mandi_res'),
    path( '/farmerprofile', views.farmerpro, name='farmer_profile')   
]
