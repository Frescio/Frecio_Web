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
from user.models import location
import matplotlib.pyplot as plt
import numpy

from django.core.files.storage import FileSystemStorage


img_height, img_width=64,64
# Create your views here.
API_KEY = 'ca2fddc268564e208b6f7596225531d2'

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


dis_dic = {
        'Apple__Apple_scab': 

['To get rid of Apple Scab, consider the following suggestions: ',

'1. Using a fungicide to treat apple scab may be necessary.',
'2. some of the best apple scab fungicide options include Bonide Captan, summer lime sulfur, Spectracide Immunox and wettable sulfur. All these apple scab fungicide options are best applied when the weather is warm. Wait until temperatures are higher than 60 degrees and the blooms or leaves of the tree are wet.'
],
    'Apple__Black_rot': 
[
'To get rid of apple black rot, consider the following suggestions:',

'1. Cultural practices and fungicides can help control sooty blotch and flyspeck.  ',
'2. Proper pruning of apples trees and thinning of fruit promote drying and help reduce disease severity.  ',
'3. Fungicides may also be necessary. ',
'4. If control measures fail, sooty blotch and flyspeck can be removed with vigorous rubbing.',
],

    'Apple___Cedar_apple_rust':
[

'To get rid of apple rust, consider the following suggestions:',

'1. Tolerate leaf spots and fruit infection on apples, hawthorns, serviceberry and other plants in the Rosaceae family. They do not significantly affect the health of the tree if leaf loss does not occur.',
'2. Do not plant eastern red cedar and juniper plants within a few hundred yards of susceptible Rosaceae plants. This will reduce diseases problems but not completely eliminate them.',
'3. Inspect nearby juniper and red cedar trees in late winter or early spring. Prune and remove brown, woody galls found before orange, gelatinous structures form in the spring. This will help reduce the level of infection on nearby Rosaceae plants.',
'4. Prune and remove infected twigs or branches on Rosaceae plants if they occur.',
],
    
   'Apple___healthy': 


['The apple leaf is healthy.']
,

   'Blueberry___healthy': 
[
'The bluberry leaf is healthy.']
,

    'Cherry_(including_sour)___Powdery_mildew':
[
'To get rid of cherry powdery mildew, consider the following suggestions:',

'1. Fungicide applications should be started two weeks after bloom when leaves are completely unfolded.',
'2. Fungicides with an active ingredient of myclobutanil or captan will protect leaves from infection with cherry leaf spot when applied properly.'
],
    'Cherry_(including_sour)___healthy': 

['The cherry leaf is healthy.']
,

'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 


['To get rid of corn gray leaf spot, consider the following suggestions:',

'1. During the growing season, foliar fungicides can be used to manage gray leaf spot outbreaks.',
'2. Farmers must consider the cost of the application and market value of their corn before determining if fungicides will be an economical solution to GLS.',
]
#,
#     'Corn_(maize)___Common_rust_':


# 1. To get rid of corn common rust, consider the following suggestions:
# 2. Use a mild bleach solution, and rinse and dry them thoroughly before putting them away. Rake up and remove fallen or dead leaves and other plant debris. Some gardeners compost vegetation that show signs of rust, expecting the compost pile to heat up enough to kill the spores.
# """ ,

#     'Corn_(maize)___Northern_Leaf_Blight': #rice
# """

# To get rid of corn leaf blight, consider the following suggestions:

# 1. Foliar fungicides may be applied early in the growing season to corn seedlings as a risk-management tool for northern corn leaf blight and other corn diseases, including anthracnose leaf blight and corn eyespot. 
# 2. After northern corn leaf blight is identified during the growing season, fungicides should be applied early in the disease outbreak for maximum effectiveness.
# 3. Crop rotation remains a solid tactic to help diminish disease threats. Rotating from corn to non-host crops helps reduce favorable environmental conditions for disease pathogens, risk of infection and disease levels. 
# """
# , 
#     'Corn_(maize)___healthy':
# """
# The corn leaf is healthy.
# """
# , 'Grape___Black_rot':
# """

# To get rid of grape black rot, consider the following suggestions:

# 1.  Mancozeb, and Ziram are all highly effective against black rot. Because these fungicides are strictly protectants, they must be applied before the fungus infects or enters the plant. They protect fruit and foliage by preventing spore germination. They will not arrest lesion development after infection has occurred.
# 2. Copper products can be effective if applied at the right time, and some are approved for organic systems. 
# 3. Probiotic sprays, plant defence-enhancing sprays or sprays containing antagonistic bacteria such as Bacillus spp. may be successful in controlling black rot.
# """
# , 'Grape___Esca_(Black_Measles)':
# """

# To get rid of corn grape ecsa, consider the following suggestions:

# 1. Prune out any and all areas with lesions; grapevines can handle a severe pruning — when in doubt, cut it out.
# 2. Prevention is key when dealing with grape black rot. During your fall clean-up, make sure that all mummies have been removed from the vine and all plant material on the ground below is destroyed.
# """
# , 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)':
# """

# To get rid of grape leaf blight, consider the following suggestions:

# 1. Spraying of the grapevines at 3-4 leaf stage with fungicides like Bordeaux mixture @ 0.8% or Copper Oxychloride @ 0.25% or Carbendazim @ 0.1% are effective against this disease.
# 2. Horticultural mineral oils will denature the fungus since most of its mycelia reside on the outside of grape tissues. Apply a 1-2% oil treatment to the vines within 2 days of the water application.
# """
#     , 'Grape___healthy':
# """
# The grape leaf is healthy.
# """
# , 'Orange___Haunglongbing_(Citrus_greening)':
# """

# To get rid of citrus greening, consider the following suggestions:

# 1. Currently there is no cure for citrus greening disease; the best control strategy is to keep healthy plants from being infected.
# 2. One of the most effective ways to prevent the disease is to avoid moving plants and plant materials from areas under regulatory quarantine or where the insect or disease is present.
# """
#     , 'Peach___Bacterial_spot':
# """

# To get rid of peach bacterial spot, consider the following suggestions:

# 1. Compounds available for use on peach and nectarine for bacterial spot include copper, oxytetracycline (Mycoshield and generic equivalents), and syllit+captan; however, repeated applications are typically necessary for even minimal disease control.
# 2. Remove symptomatic plants from the field or greenhouse to prevent the spread of bacteria to healthy plants.
# 3. Burn, bury or hot compost the affected plants and DO NOT eat symptomatic fruit.
# """
#         , 'Peach___healthy':
# """
# The peach leaf is healthy.
# """
#         , 'Pepper,_bell___Bacterial_spot':
# """

# To get rid of peach bacterial spot, consider the following suggestions:

# 1. Compounds available for bacterial spot include copper, oxytetracycline (Mycoshield and generic equivalents), and syllit+captan; however, repeated applications are typically necessary for even minimal disease control.
# 2. Remove symptomatic plants from the field or greenhouse to prevent the spread of bacteria to healthy plants.
# 3. Burn, bury or hot compost the affected plants and DO NOT eat symptomatic fruit.
# """
# , 'Pepper,_bell___healthy':
# """
# The pepper bell leaf is healthy.
# """
#     , 'Potato___Early_blight':
# """

# To get rid of potato early blight, consider the following suggestions:

# 1. Treatment of early blight includes prevention by planting potato varieties that are resistant to the disease.
# 2. Late maturing are more resistant than early maturing varieties. 
# 3. Avoid overhead irrigation and allow for sufficient aeration between plants to allow the foliage to dry as quickly as possible.
# """
#     , 'Potato___Late_blight':
# """

# To get rid of potato late blight, consider the following suggestions:

# 1. The severe late blight can be effectively managed with prophylactic spray of mancozeb at 0.25% followed by cymoxanil+mancozeb or dimethomorph+mancozeb at 0.3% at the onset of disease and one more spray of mancozeb at 0.25% seven days after application of systemic fungicides.
# 2. Late blight is controlled by eliminating cull piles and volunteer potatoes, using proper harvesting and storage practices, and applying fungicides when necessary. 
# """
# , 'Potato___healthy':
# """
# The potato leaf is healthy.
# """
#     , 'Raspberry___healthy':
# """
# The raspberry leaf is healthy.
# """
#  , 'Soybean___healthy':
# """
# The soyabean leaf is healthy.
# """
# , 'Squash___Powdery_mildew':
# """

# To get rid of squash powdery mildew, consider the following suggestions:

# 1. A better treatment solution for your squash plants is baking soda.
# 2. Provide good air circulation by spacing squash plants several feet apart.
# """
# , 'Strawberry___Leaf_scorch':
# """

# To get rid of strawberry leaf scorch, consider the following suggestions:

# 1. If the disease is detected early, its development may be slowed using fungicides.
# 2. Removing those dead leaves can clear out spores or other items on the leaves. That can spread disease to your strawberries or make them mold. And removing the dead leaves, removes homes for slugs and other pest.
# """ 
#      , 'Strawberry___healthy':
# """
# The strawberry leaf is healthy.
# """
#     , 'Tomato___Bacterial_spot':
# """

# To get rid of tomato bacterial spot, consider the following suggestions:

# 1. Remove symptomatic plants from the field or greenhouse to prevent the spread of bacteria to healthy plants. 
# 2. Burn, bury or hot compost the affected plants and DO NOT eat symptomatic fruit."""
    
# , 'Tomato___Early_blight':
# """

# To get rid of tomato early blight, consider the following suggestions:

# 1. Thoroughly spray the plant (bottoms of leaves also) with Bonide Liquid Copper Fungicide concentrate or Bonide Tomato & Vegetable.
# 2. Destroy infested plants by burning or burying them. Rotate vegetables to different parts of your garden each year to avoid areas where infested debris (and thus spores of Alternaria solani) may be present."""   

#     , 'Tomato___Late_blight':
# """

# To get rid of tomato late blight, consider the following suggestions:

# 1. Baking soda has fungicidal properties that can stop or reduce the spread of early and late tomato blight. Baking soda sprays typically contain about 1 teaspoon baking soda dissolved into 1 quart of warm water. Adding a drop of liquid dish soap or 2 1/2 tablespoons vegetable oil helps the solution stick to your plant.
# 2. If blight is detected on shoots or leaves, remove them from the plant and dispose of diseased foliage off-site."""
   
# , 'Tomato___Leaf_Mold':
# """

# To get rid of tomato leaf mold, consider the following suggestions:

# 1. Use fans and avoid overhead watering to minimize leaf wetness.
# 2. Also, stake and prune plants to increase ventilation. If the disease is detected, apply a fungicide according to the manufacturer's instructions at the first sign of infection.
# """
# , 'Tomato___Septoria_leaf_spot':
# """

# To get rid of tomato septoria leaf spot, consider the following suggestions:

# 1. Remove diseased leaves. If caught early, the lower infected leaves can be removed and burned or destroyed. 
# 2. Improve air circulation around the plants. If the plants can still be handled without breaking them, stake or cage the plants to raise them off the ground and promote faster drying of the foliage."""

# , 'Tomato___Spider_mites Two-spotted_spider_mite':
# """

# To get rid of tomato two spotted spider mite, consider the following suggestions:

# 1. The rubbing alcohol you have around the house also can kill spider mites. Soak cotton balls in rubbing alcohol and wipe across the foliage of infested houseplants.
# 2. Make sure that all plants get the necessary amount of sunlight for the variety.
# """
# , 'Tomato___Target_Spot':
# """

# To get rid of tomato target spot, consider the following suggestions:

# 1. Remove a few branches from the lower part of the plants to allow better airflow at the base.
# 2. Remove and burn the lower leaves as soon as the disease is seen, especially after the lower fruit trusses have been picked.
# """
# , 'Tomato___Tomato_Yellow_Leaf_Curl_Virus':
# """

# To get rid of tomato yellow leaf curls virus spot, consider the following suggestions:

# 1. Rogue tomato or pepper plants with early symptoms of TYLCV can be removed from fields by placing infected-looking plants in plastic bags immediately at the beginning season, especially during first three to four weeks.
# 2. Maintain good weed control in the field and surrounding areas.
# """
# , 'Tomato___Tomato_mosaic_virus':
# """

# To get rid of tomato target spot, consider the following suggestions:

# 1. Remove all infected plants and destroy them. Do NOT put them in the compost pile, as the virus may persist in infected plant matter. 
# 2. Monitor the rest of your plants closely, especially those that were located near infected plants.
# """
#     , 'tomato___healthy':
# """
# The', tomato leaf is healthy.
# """
}


def home(request):

    url = f'https://newsapi.org/v2/everything?q=kharif + crop&from=2021-11-27&sortBy=publishedAt&apiKey={API_KEY}'
    response = requests.get(url)
    data = response.json()
    context = {}
    articles = []
    try:
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
    except:
        # articles = []
        context = {}
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
    model = keras.models.load_model('home/models/Trained_model.h5')
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
    
import torch
import torch.nn as nn           # for creating  neural networks
from torch.utils.data import DataLoader # for dataloaders 
from PIL import Image           # for checking images
import torch.nn.functional as F 


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available:
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_default_device()
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
# def predict_image(img, model):
#     """Converts image to array and return the predicted class
#         with highest probability"""
#     # Convert to a batch of 1
#     xb = to_device(img.unsqueeze(0), device)
#     # Get predictions from model
#     yb = model(xb)
#     # Pick index with highest probability
#     _, preds  = torch.max(yb, dim=1)
#     # Retrieve the class label

#     return train.classes[preds[0].item()]     


def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)









# for calculating the accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# base class for the model
class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                   # Generate prediction
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)          # Calculate accuracy
        return {"val_loss": loss.detach(), "val_accuracy": acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()       # Combine loss  
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy} # Combine accuracies
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))
# resnet architecture 
class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True) # out_dim : 128 x 64 x 64 
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True) # out_dim : 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True) # out_dim : 512 x 4 x 44
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))
        
    def forward(self, xb): # xb is the loaded batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out        

from PIL import Image
from torchvision.transforms import ToTensor

dis_class=['Tomato___Late_blight', 'Tomato___healthy', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Potato___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Tomato___Early_blight', 'Tomato___Septoria_leaf_spot', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Strawberry___Leaf_scorch', 'Peach___healthy', 'Apple___Apple_scab', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Bacterial_spot', 'Apple___Black_rot', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Peach___Bacterial_spot', 'Apple___Cedar_apple_rust', 'Tomato___Target_Spot', 'Pepper,_bell___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___Late_blight', 'Tomato___Tomato_mosaic_virus', 'Strawberry___healthy', 'Apple___healthy', 'Grape___Black_rot', 'Potato___Early_blight', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Common_rust_', 'Grape___Esca_(Black_Measles)', 'Raspberry___healthy', 'Tomato___Leaf_Mold', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Pepper,_bell___Bacterial_spot', 'Corn_(maize)___healthy']


def disease_result(request):
    context = {}
    print("zzzzzzzzzz")
    if request.POST.get('submit') == "predict_disease":
        
        fileObj = request.FILES['dis_photo']
        # fileObj=request.FILES['filePath']
        fs=FileSystemStorage()
        filePathName=fs.save(fileObj.name,fileObj)
        filePathName=fs.url(filePathName)
        testimage='.'+filePathName
        img = image.load_img(testimage)

        model_disease=ResNet9(3,38)
        model_disease.load_state_dict(torch.load("home/models/plant-disease-model.pb",map_location=torch.device('cpu')))
        model_disease.eval()
        # model_disease=torch.load("home/models/plant-disease-model-complete.pth")
        # model_disease.eval()
        # Convert to a batch of 1
        xb=ToTensor()(img).unsqueeze(0) 

        # xb = to_device(img.unsqueeze(0), device)
        # Get predictions from model
        yb = model_disease(xb)
        _, preds  = torch.max(yb, dim=1)
        # Retrieve the class label

        res=dis_class[preds[0].item()]
        print("xxxxxxxxxxxxxxxxx   ",res)
        solutions=dis_dic['Apple__Apple_scab']
        context['result'] = 'Apple__Apple_scab'
        context['recommendations'] = solutions
        print(solutions)
    return render( request, 'home/disease_result.html',context)


def disease_pred(request):
    return render( request, 'home/disease_predict.html')


def profile(request):
    if request.POST:
        first = request.POST['first']
        last = request.POST['last']
        ustate = request.POST['state']
        ucity = request.POST['district']
        # User.objects.filter(user=request.user).update(first_name=first, last_name=last)
        try:
            address = location.objects.get(user=request.user)
            location.objects.filter(user=request.user).update( state=ustate, city=ucity)
        except:
            address = location( user=request.user, state=ustate, city=ucity )
            address.save()

    address = location.objects.filter(user=request.user)
    context = {'location':address}
    print("aaaaaaa")
    print(address)
    return render( request, 'home/profile.html', context)

def edit_profile(request):
    context={}
    ustate = " "
    ucity = " "
    data2=[]
    # data = pd.read_csv("fertilizer.csv")
    with open('home/state.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0]!="All":
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

    try:
        address = location.objects.get(user=request.user)
        
        print("bbbbb")
        print(address)
        print("Cccc")
        print(address.state)
        ustate = address.state
        ucity = address.city
        context['ustate'] = ustate
        context['ucity'] = ucity    
        return render( request, 'home/edit_profile.html', context)
    
    except:
    
        context['ustate'] = "NA"
        context['ucity'] = "NA"
        print(context['state'])
        print("FFFFFFFFFF")
        # print(context['city'])
        return render( request, 'home/edit_profile.html', context)


def chatbot_index(request):
    return render( request, 'home/chatbot.html')



from django.http.response import JsonResponse
from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.contrib.auth.models import User, auth
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from tensorflow.python.framework.tensor_conversion_registry import get
import numpy
import pickle
import json
import random
import numpy as np
from django.forms.models import model_to_dict

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.python.keras.saving.saved_model.utils import list_all_layers

import speech_recognition as sr

def speech_to_text(request):
    recognizer = sr.Recognizer()
    print("lol")
    with sr.Microphone() as mic:
        print("listening...")
        audio = recognizer.listen(mic)
    try:
        print("trying1...")
        text = recognizer.recognize_google(audio)
        print("trying2...",text)
        return  HttpResponse(text)
    except:
        print("except...")
        return   HttpResponse("ERROR")
# print(speech_to_text())



def chatbot(request):
    message = request.GET.get('msg')
    lemmatizer = WordNetLemmatizer()

    intents= json.loads(open('home/models/intents.json').read(), strict=False)

    words = pickle.load(open('home/models/words.pkl','rb'))
    classes = pickle.load(open('home/models/classes.pkl','rb'))
    # model = load_model('chatbot_model.model')
    model = load_model('home/models/chatbotmodel.h5')

    def clean_up_sentence(sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words

    def bag_of_words(sentence):
        sentence_words = clean_up_sentence(sentence)
        bag = [0]*len(words)
        for w in sentence_words:
            for i,word in enumerate(words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(sentence):
        bow = bag_of_words(sentence)
        res = model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key= lambda x: x[1], reverse = True)
        return_list = []
        for r in results:
            return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
        return return_list

    def get_response(intents_list,intents_json):
        try:
            tag = intents_list[0]['intent']
        except:
            return "Sorry I can't understand" 
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result

    print("GO! Bot is running!")


    
    ints = predict_class(message)
    res = get_response(ints,intents)
    print(res)
    return HttpResponse(res)
