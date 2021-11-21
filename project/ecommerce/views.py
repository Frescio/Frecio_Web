from django.http.response import HttpResponse
from django.shortcuts import render
import csv

from user.models import crop
from user.models import wishlist
from django.contrib import messages
# Create your views here.
from django.core import serializers


def mycrops(request):
    if request.user.is_authenticated:
        if request.user.isFarmer:
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
            messages.info(request, 'Please Register as a Farmer to view this page.')
            return render( request, 'ecommerce/error_msg.html')
            # return HttpResponse("Please Register as a Farmer to upload your crops")
    else:
        messages.info(request, 'Please Login/Register as a Farmer to view this page.')
        return render( request, 'ecommerce/error_msg.html')
    

def viewcrops(request):
    # return( request, 'ecommerce/view_your_added_crops.html')
    
    if request.user.is_authenticated:
        all_crops = crop.objects.filter()
        # all_crops = crop.objects.raw("select * from user_crop where user_crop.id != user_wishlist.crop")
        # print("xxxxxxxxxxxxxxxxxx",all_crops2,all_crops)
        if request.POST:
            crop_id = request.POST['crop_id']
            wished = wishlist( user=request.user, crop = crop.objects.get(pk=crop_id) )
            wished.save()
        # for p in crop.objects.raw('SELECT * FROM myapp_crop'):
        #     print(p)
        wish_list = wishlist.objects.filter(user=request.user)
        # wish_list = serializers.serialize("json", wishlist.objects.filter(user=request.user))
        print(wish_list)
        for wish in wish_list:
            # print(wish)
            # cropp = crop.objects.get()
            # print(wish.crop.id)
            all_crops = all_crops.exclude(pk=wish.crop.id)
        # for pair in wish_list:
        #     if pair.crop in all_crops:

        return render( request, 'ecommerce/view_all_crops.html' , { 'all_crops':all_crops, 'wished_crop':wish_list} )
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
