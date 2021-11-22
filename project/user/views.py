from django.http.response import HttpResponse
from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from .forms import SignupForm, LoginForm
from django.contrib.auth import authenticate, login, logout
from django.urls import reverse
from datetime import datetime
from home import views
# from django.core.urlresolvers import reverse_lazy
from django.contrib import messages
from .models import *


def register(request):
    context = {}
    context['login_form'] = LoginForm()
    context['signup_form'] = SignupForm()
    if request.method == "POST":
        if request.POST.get('submit') == 'login':
            if request.user.is_authenticated:
                return redirect('')
            if request.POST:
                form = LoginForm(request.POST)
                if form.is_valid():
                    phone = request.POST['phone']
                    password = request.POST['password']
                    user = authenticate(phone=phone, password=password)
                    if user:
                        login(request, user)
                        return redirect(reverse('homepage'))
                else:
                    context['login_form'] = form
                
            return render(request, 'user/register.html', context)
        elif request.POST.get('submit') == 'signup':
            print("xxxxxx")
            if request.POST:
                form = SignupForm(request.POST)
                if form.is_valid():
                    print("ppppppp")
                    form.save()
                    phone = form.cleaned_data.get('phone')
                    raw_pass = form.cleaned_data.get('password1')
                    new_account = authenticate(phone=phone, password=raw_pass)
                    login(request, new_account)
                    return redirect(reverse('homepage'))
                    
                else:
                    context['signup_form'] = form

    return render(request, 'user/register.html', context)
        

def usrlogout(request):
    logout(request)
    return redirect('/')

def profile(request):
    
    return render(request, 'user/profile.html')