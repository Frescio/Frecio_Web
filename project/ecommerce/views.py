from django.http.response import HttpResponse
from django.shortcuts import render
import csv
import requests

from user.models import crop

from user.models import wishlist
# Create your views here.

def mycrops(request):
    if request.user.is_authenticated:
        if request.POST:
            print("bbbbbbb")
            # name = request.POST.get('name')
            if request.POST.get('submit') == "add_new_crop":
                crop_name = request.POST['crop_name']
                price = request.POST['price']
                quantity = request.POST['quantity']
                photo = request.FILES['photo']
                new_crop = crop( user=request.user, crop_name = crop_name, price=price, quantity=quantity, photo=photo)
                new_crop.save()
                print(new_crop)
            # elif request.POST.get('submit') == "edit_crop":

            #     return 
            elif request.POST.get('submit') == "delete_crop":
                print("asdfsad")
                crop.objects.filter(id=request.POST.get('crop_id')).delete()

            elif request.POST.get('submit') == "edit_crop":
                
                crop_name = request.POST['crop_name']
                price = request.POST['price']
                quantity = request.POST['quantity']
                photo = request.FILES['photo']
                crop.objects.filter(id=request.POST.get('crop_id')).update(crop_name=crop_name,price=price,quantity=quantity,photo=photo)
                
                # return 
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

        crops_added = crop.objects.filter(user=request.user)
        
        return render( request, 'ecommerce/view_your_added_crops.html', {'crop':data,'crops_added': crops_added })
    else:
        return HttpResponse("sorry")
    

def viewcrops(request):
    # return( request, 'ecommerce/view_your_added_crops.html')
    
    if request.user.is_authenticated:
        all_crops = crop.objects.filter()
        if request.POST:
            crop_id = request.POST['crop_id']
            wished = wishlist( user=request.user, crop = crop.objects.get(pk=crop_id) )
            wished.save()
        wish_list = wishlist.objects.filter(user=request.user)
        
        return render( request, 'ecommerce/view_all_crops.html' , { 'all_crops':all_crops } )
    else:
        return HttpResponse("Sorry")

def mywishlist(request):

    if request.user.is_authenticated:
        wish_list = wishlist.objects.filter(user=request.user)
        
        # if request.POST:
        #     if request.POST.get('submit') == "add_to_wishlist":
        #         crop_id = request.POST.get('crop_id')
        #         wished = wishlist( user=request.user, crop = crop.objects.get(pk=crop_id) )
        #         wished.save()
                
        if request.POST.get('submit') == "remove_crop":
                print("asdfsad")
                wishlist.objects.filter(id=request.POST.get('wish_id')).delete()
        return render( request, 'ecommerce/list_of_wish.html', { 'wishlist':wish_list } )
    else:
        return HttpResponse("Sorry")

def sellcrops(request):
    print("bbbbbbb")
    if request.user.is_authenticated:
        if request.POST:
            print("bbbbbbb")
            # name = request.POST.get('name')
            if request.POST.get('submit') == "add_new_crop":
                crop_name = request.POST['crop_name']
                price = request.POST['price']
                quantity = request.POST['quantity']
                photo = request.FILES['photo']
                new_crop = crop( user=request.user, crop_name = crop_name, price=price, quantity=quantity, photo=photo)
                new_crop.save()
                print(new_crop)
            # elif request.POST.get('submit') == "edit_crop":

            #     return 
            elif request.POST.get('submit') == "delete_crop":
                print("asdfsad")
                crop.objects.filter(id=request.POST.get('crop_id')).delete()

            elif request.POST.get('submit') == "edit_crop":
                
                crop_name = request.POST['crop_name']
                price = request.POST['price']
                quantity = request.POST['quantity']
                photo = request.FILES['photo']
                crop.objects.filter(id=request.POST.get('crop_id')).update(crop_name=crop_name,price=price,quantity=quantity,photo=photo)
                
                # return 
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

        crops_added = crop.objects.filter(user=request.user)
        all_crops = crop.objects.filter()
        return render( request, 'ecommerce/sell.html', {'crop':data, 'crops_added': crops_added ,"all_crops":all_crops })
    else:
        return HttpResponse("sorry")


def mandi(request):
    context = {}
    # context['crop'] = {'sdf','asd','lol'}
    data1=[]
    # data = pd.read_csv("fertilizer.csv")
    with open('home/commodity.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data1.append(row[0])
    data1.sort()
    context['commodity'] = data1


    data2=[]
    # data = pd.read_csv("fertilizer.csv")
    with open('home/state.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data2.append(row[0])
    data2.sort()
    context['state'] = data2

    data3=[]
    # data = pd.read_csv("fertilizer.csv")
    with open('home/district.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data3.append(row[0])
    data3.sort()
    context['district'] = data3



    return render( request, 'ecommerce/mandi.html',context)

def mandipred(request):
    if request.POST:
        commodity = request.POST['commodity']
        state = request.POST['state']
        district= request.POST['district']


        print(state)
        print(district)
        print(commodity)
        #PH = request.POST['ph']
        #crop_name = request.POST['cropname']
        #data=pd.read_csv('home/fertilizer.csv')
        # data.drop("Unnamed: 0",axis=1,inplace=True)
        # n_actual = data[data['Crop'] == crop_name]['N'].iloc[0]
        #p_actual = data[data['Crop'] == crop_name]['P'].iloc[0]
        #k_actual = data[data['Crop'] == crop_name]['K'].iloc[0]
        #p_actual=  data[data['Crop'] == crop_name]['pH'].iloc[0]

        #n = n_actual - int(N)
        #p = p_actual - int(P)
        #k = k_actual - int(K)
        #ph = p_actual - int(PH)
        
        #nkey=""
        #pkey=""
        #kkey=""
        #phkey=""
        #if n < -10:
            #nkey = "NHigh"
        #elif n>10:
            #nkey = "Nlow"
        #else:
           # nkey="Nok"
            
        #if p < -10:
           # pkey = "PHigh"
        #elif p>10:
           # pkey = "Plow"
        #else:
            #pkey="Pok"

        #if k < -10:
            #kkey = "KHigh"
        #elif k>10:
            #kkey = "Klow"
        #else:
            #kkey="Kok"
            
        #if ph<-10:
           # phkey="phHigh"
        #elif ph>10:
            #phkey="phLow"
        #else:
            #phkey="phok"
        
        
        #context['nkey'] = nkey
        #context['pkey'] = pkey
        #context['kkey'] = kkey
        #context['phkey'] = phkey

        #url = f'https://newsapi.org/v2/top-headlines?country={country}&apiKey={API_KEY}'
        url=f'https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key=579b464db66ec23bdd00000105e2ba56e600420b6fbbdfd39f5af462&format=json&offset=0&limit=1000'
        response = requests.get(url)
        data = response.json()
        records = data['records']
        final =[]

        for i in records:
            if i['commodity'] == commodity:
                if i['state'] == state:
                    if i['district'] == district:
                        final.append(i)
                    elif district == 'All':
                        final.append(i)    
                
                elif state == 'All':
                    final.append(i)        
                
            elif commodity == 'All':
                if i['state'] == state:
                    if i['district'] == district:
                        final.append(i)
                    elif district == 'All':
                        final.append(i)
        
                elif state == 'All':
                    final.append(i)

        records=final  

        print(len(records))
        a=len(records)
        if a !=0:
            context = {'records' : records}
            return render( request, 'ecommerce/mandi_result_test.html', context)

        else:
            #number="empty"
            #context = {'number' : number}
            return render( request, 'ecommerce/mandi_result_sorry.html')    
