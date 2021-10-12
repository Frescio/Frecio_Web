from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.db.models import fields
from django.contrib.auth import authenticate
from .models import User

class SignupForm(UserCreationForm):
    phone = forms.CharField(help_text="A valid phone no. id is required")

    class Meta:
        model = User
        fields = ('phone', 'first_name', 'last_name', 'password1', 'password2')

class LoginForm(forms.ModelForm):
    password = forms.CharField(label='Password', widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ('phone', 'password')

    def clean(self):
        if self.is_valid():
            phone = self.cleaned_data['phone']
            password = self.cleaned_data['password']
            if not authenticate(email=email, password=password):
                raise forms.ValidationError("Invalid Login Credentials")
