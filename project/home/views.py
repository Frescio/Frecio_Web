from django.http.response import HttpResponse, HttpResponseNotModified
from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.core.mail import send_mail
from django.contrib import messages
import pandas as pd
import csv
import requests

# Create your views here.
API_KEY = 'd0b69496c18e463f888a273cb521ea9f'

def home(request):
    url = f'https://newsapi.org/v2/everything?q=kharif + crop&from=2021-10-29&sortBy=publishedAt&apiKey={API_KEY}'
    response = requests.get(url)
    data = response.json()
    articles = data['articles']
        
    def smart_truncate(content, length=55, suffix='...'):
        if len(content) <= length:
            return content
        else:
            return ' '.join(content[:length+1].split(' ')[0:-1]) + suffix

    for x in range(4):
         string=articles[x]['title']
         string=smart_truncate(string)
         articles[x]['title']=string

    context = {
        'articles0' : articles[0], 'articles1' : articles[1],'articles2' : articles[2],'articles3' : articles[3]
    }
    return render( request, 'home/homepage.html', context )
    #return render( request, 'home/homepage.html' )
    # return render(request, 'home/base.html' )

def register(request):
    return render( request, 'home/register.html' )

def about(request):
    return render( request, 'home/about.html')

def contact(request):
    if request.POST:
        name = request.POST['Name']
        email =  request.POST['Email']
        phone = request.POST['Telephone']
        message = request.POST['Message']
        send_mail(
            'New Message Received',
            "Message: {}\n\nFrom {}\nEmail: {}\nContact No: {}".format(message,name, email, phone),
            '',
            ['frescio.farm@gmail.com'],
            fail_silently=False,
        )
        messages.success(request, 'Thank you for writing to us, we will contact you shortly!')
        return redirect('contactus')
    return render( request, 'home/contact.html')
    

def fertilizer(request):
    context = {}
    # context['crop'] = {'sdf','asd','lol'}
    data=[]
    # data = pd.read_csv("fertilizer.csv")
    with open('home/fertilizer.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row[1])
    data.sort()
    context['crop'] = data

    return render( request, 'home/fertilizer.html',context)

def fertpred(request):
    if request.POST:
        N = request.POST['nitrogen']
        P = request.POST['phosphorous']
        K = request.POST['pottasium']
        PH = request.POST['ph']
        crop_name = request.POST['cropname']
        data=pd.read_csv('home/fertilizer.csv')
        # data.drop("Unnamed: 0",axis=1,inplace=True)
        n_actual = data[data['Crop'] == crop_name]['N'].iloc[0]
        p_actual = data[data['Crop'] == crop_name]['P'].iloc[0]
        k_actual = data[data['Crop'] == crop_name]['K'].iloc[0]
        p_actual=  data[data['Crop'] == crop_name]['pH'].iloc[0]

        n = n_actual - int(N)
        p = p_actual - int(P)
        k = k_actual - int(K)
        ph = p_actual - int(PH)
        
        nkey=""
        pkey=""
        kkey=""
        phkey=""
        if n < -10:
            nkey = "NHigh"
        elif n>10:
            nkey = "Nlow"
        else:
            nkey="Nok"
            
        if p < -10:
            pkey = "PHigh"
        elif p>10:
            pkey = "Plow"
        else:
            pkey="Pok"

        if k < -10:
            kkey = "KHigh"
        elif k>10:
            kkey = "Klow"
        else:
            kkey="Kok"
            
        if ph<-10:
            phkey="phHigh"
        elif ph>10:
            phkey="phLow"
        else:
            phkey="phok"
        
        context = {}
        context['nkey'] = nkey
        context['pkey'] = pkey
        context['kkey'] = kkey
        context['phkey'] = phkey

        return render( request, 'home/fertilizer_result.html', context)
