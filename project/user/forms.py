from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.db.models import fields
from django.contrib.auth import authenticate
from .models import User
from django.utils.translation import ugettext_lazy as _
from phonenumber_field.formfields import PhoneNumberField

class SignupForm(UserCreationForm):
    # phone = forms.CharField(help_text="A valid phone no. id is required")
    phone = PhoneNumberField()
    class Meta:
        model = User
        fields = ('phone', 'first_name', 'last_name', 'isFarmer', 'location', 'password1', 'password2')
        labels = {
            'isFarmer': _('Do you want to register as a Farmer?'),
        }

class LoginForm(forms.ModelForm):
    password = forms.CharField(label='Password', widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ('phone', 'password')

    def clean(self):
        if self.is_valid():
            phone = self.cleaned_data['phone']
            password = self.cleaned_data['password']
            if not authenticate(phone=phone, password=password):
                raise forms.ValidationError("Invalid Login Credentials")
