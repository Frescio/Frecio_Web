from django.http.response import HttpResponse, HttpResponseNotModified
from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.core.mail import send_mail
from django.contrib import messages
import pandas as pd
import csv
import requests
from keras.preprocessing import image
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import numpy

from django.core.files.storage import FileSystemStorage
img_height, img_width=64,64
# Create your views here.
API_KEY = 'd0b69496c18e463f888a273cb521ea9f'

pest_dic = {
        'aphids': 


['To get rid of aphids, consider the following suggestions: ',

'1. Try spraying cold water on the leaves, sometimes all aphids need is a cool blast to dislodge them.',
'2. If you have a large aphid invasion, dust plants with flour. It constipates the pests. ',
'3. Neem oil, insecticidal soaps, and horticultural oils are effective against aphids.',
'4. You can often get rid of aphids by wiping or spraying the leaves of the plant with a mild solution of water and a few drops of dish soap. Soapy water should be reapplied every 2-3 days for 2 weeks.',
'5. One variation of this soap-water mix includes cayenne pepper: Stir together 1 quart water, 1 tsp liquid dish soap, and a pinch of cayenne pepper. Do not dilute before spraying on plants.',
'6. Diatomaceous earth (DE) is a non-toxic, organic material that will kill aphids. Do not apply DE when plants are in bloom; it is harmful to pollinators, too.'],
 
    'armyworm': 
[
'To get rid of armyworms, consider the following suggestions:',

'1. Release trichogramma wasps to parasitize any newly laid eggs. These tiny beneficial insects — 1mm or less — insert their eggs inside of pest eggs, killing them before they enter the plant-eating larval stage.',
'2. Other beneficial insects, such as lacewing, ladybugs and minute pirate bugs feed on armyworm eggs as well as the young larval stage. Remember: beneficial ins',
'3. Plant to attract birds and beneficial insects. In the fall, uncover and turn your soil before putting it to bed, giving birds a chance to pick off the exposed pupae.',
'4. Applications of Garden Dust (Bt-kurstaki) or OMRI-listed Monterey Garden Insect Spray (spinosad) will kill armyworms.',
'5. After the season has advanced, natural horticultural oil sprays can be used on plants showing signs of worm infestations. Multi-purpose neem oil spray is effective on various stages of the larvae as well as mites. It also prevents fungus growth. Complete coverage, including undersides of leaves and junctions with stems, is critical.',
'6. Use fast-acting organic insecticides if pest levels become intolerable.']
,

    'beetle':
[

'To get rid of beetles, consider the following suggestions:',

'1. Use water and dish soap. This method is effective, non toxic and kills beetles fast.',
'2. Vacuum beetles up. This method is effective, non toxic and safe.',
'3. Hang beetle traps.',
'4. Use insecticidal soap on bushes and landscaping. ']
,
    
   'bollworm': 
[

'To get rid of bollworms, consider the following suggestions:',

'1. Treat an infestation only when ten eggs or 5 small worms per a hundred cotton plants are present in late July to early August. ',
'2. Handpick and destroy eggs and small bollworms. This is feasible in small plots or when infestations are low. ',
'3. Plough the soil after harvesting. This exposes pupae, which may then be killed by natural enemies or through desiccation by the sun.',
'4. Garlic is reported to be effective against bollworm on cotton and maize.']
,

   'earthworm': 
[

'To get rid of earthworms, consider the following suggestions:',

'1. Proper Maintenance. Proper maintenance and care for your lawn is the most effective way of getting the infestation under control.',
'2. Pick Them Up. One surefire but time-consuming way of removing them your property is to remove them manually.',
'3. Change the pH of the Soil. Worms do not like acidic soil.  Applying iron sulfate (found in most lawn and garden stores) every 8 to 10 weeks will keep your soil on the acidic side.  ',
'4. Introduce Predatory Species. A natural form of pest control is to introduce a predator species. Some grub worms will feed on earthworms in your soil. ',
'5. Pesticides - Try to choose the least toxic form of pesticide available.  Also, choose a pest control chemical specific to worms instead of using one designed to kill a broad range of pests. ']
,

    'grasshopper':
[

'To get rid of grasshopper, consider the following suggestions:',

'1. Organic grasshopper repellent spray. Grasshoppers hate the smell of cayenne pepper, garlic, and onion. Mixing both substances with water and making a spray is the greenest solution you can make.Spray your plants thoroughly and make sure you cover the whole leaf (including the bottom) and the stem.',
'2. Spraying Neem oil on your plants. Apply Neem oil to your plants to repel grasshoppers and prohibit them from laying eggs on your plants.',
'3. Introducing plants that keep the grasshoppers away. Plant flowers like Lilac, Forsythia, Moss rose and Crepe Myrtle deter grasshoppers and can make a nice addition to your garden.']
,
    
    'mites': 
[

'To get rid of mites, consider the following suggestions:',

'1. Use Diatomaceous Earth (DE). When this powder is sprinkled on bug-infested surfaces and floors, it wicks away moisture from the pests, thereby dehydrating and killing them.',
'2. Mix 3 tablespoons of dish soap with a gallon of water to kill spider mites. Spray the soap solution on infested plant leaves weekly, as needed.',
'3. Soak cotton balls in rubbing alcohol and wipe across the foliage of infested plants. Let either the dish soap or rubbing alcohol sit on the plants a few hours, and then rinse the leaves thoroughly with water.']
,

'mosquito': 
[

'To get rid of mosquitos, consider the following suggestions:'

'1. Camphor is a natural home remedy that will assist in getting rid of mosquitoes.',
'2. Garlic is made up of several properties that help keep mosquitoes away. The solution made up of garlic and water will kill mosquitoes instantly. ',
'3. Mosquitoes cant stand the scent of lavender oil so you can use this to your advantage. Keep mosquitoes away by spraying lavender oil.',
'4. This is a more time-consuming mosquito repellent method, but it works well. You can place dry ice inside a container or trap and eventually, it will attract mosquitoes due to its carbon dioxide emissions. Theyll get trapped and die inside of it.']
   ,
    
    'sawfly':


['To get rid of sawflies, consider the following suggestions:',

'1. Insecticidal Soap (Potassium Salts of Fatty Acids) & Pyrethrin - The soap will penetrate the insect s shell and kill it by dehydration. Adding Pyrethrin creates the organic equivalent of a one-two knockdown punch. Pyrethrin is a nerve agent that will absorb into the insect and kill by paralysis.',
'2. Crush Larvae. Simply don a par of gloves and simply squish the larvae on leaves with needles.',
'3. Use water and dish soap. This method is effective and non toxic.',
'4. Simply use a water hose to remove sawflies from leaves.']
 ,

    'stem_borer': #rice

[
'To get rid of stem_borer, consider the following suggestions:',

'1. Introduce parasitic wasps before the eggs are laid, as wasps are borers’ natural enemy.',
'2. You can cover crops with floating row covers to prevent egg laying, drape these row covers over frames. This will protect your plants from temperature extremities.',
'3. Grow snake gourd as it is more resistant to Stem Borers.',
'4. Crop rotation is the key to get rid of stem borers. Avoid planting cucurbits in the same plot, or plant cucumbers, melons or watermelons as borers hardly threaten them.',
'5. You can cut off the infected vine and cover it with additional soil for new root growth.']


}

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

def pest_prediction(request):
    
    
    
    return render( request, 'home/pest_pred_tem.html')
    
def img_re(img):
    img = cv2.resize(img,(64, 64))     # resize image to match model's expected sizing
    return img.reshape(1,64, 64, 3)

def predict_pest(img):
    model = keras.models.load_model('home/Trained_model.h5')
    return model.predict(img)

def pest_pre_result(request):
    context = {}
    if request.POST.get('submit') == "predict_pest":
        
        fileObj = request.FILES['pest_photo']
        # fileObj=request.FILES['filePath']
        fs=FileSystemStorage()
        filePathName=fs.save(fileObj.name,fileObj)
        filePathName=fs.url(filePathName)
        testimage='.'+filePathName
        img = image.load_img(testimage)
        
        img = image.load_img(testimage, target_size=(img_height, img_width))
        x = image.img_to_array(img)
        x=x/255
        x=x.reshape(1,img_height, img_width,3)
        arr=list(predict_pest(x)[0])
        print(arr)
        maxpos = arr.index(max(arr)) 
        pest_list=['aphids','armyworm','beetle','bollworm','earthworm','grasshopper','mites','mosquito','sawfly','stem borer']
        pest_name = pest_list[maxpos]
        context['result'] = pest_name
        context['recommendations'] = pest_dic[pest_name]

        return render( request, 'home/pest_prediction_result.html',context)
    return render( request, 'home/pest_pred_tem.html')


def disease_pred(request):
    return render( request, 'home/disease_predict.html')