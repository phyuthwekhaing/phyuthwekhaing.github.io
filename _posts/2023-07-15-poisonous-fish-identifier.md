**Training a classifier to classify poisonous fishes in deep underwater**

Deep learning and I were are in on and off relationship throughout 4-5 years span since I have been in my colleage. The topic of AI always excite me and during last year of my colleage, I decided to specialize in AI. Going through difficult subjects like convolution and calculation of them in matrixes to the easier subjects like tokenization in NLP, I enjoyed every courses in my last year of my bachelor degree regardless of how my grades turned out. 

![](/images/puffer.jpg)

I also had taken a computer vision project about identifying the texts such as signs and road names in the street photographs. The project took about six months but the research itself goes on about 2-3 months, setting up everything to work in my own computer with lower level GPU and training time including setting up GPU in python wasn't really a fun experience and it took me many hours to set up a model working with a C++ and C# web interface. After the project, I had just threw away the project since it was too time consuming to set up and there were already better solutions than I did and that was when I didn't come up with my own architecture but research on different models and demonstrate what could be an application for the model and set up the small viable prototype with a working prototype. Curent learners in deep learning wouldn't believe it could take 6 months to do so when this blogpost was done in just a few days of training and setting up in the free websites like Kaggle, Hugging face and GitHub. Stay tuned to follow the rest of this post to set up your own model to deployment with a simple classification model to showcase your project in deep learning!!  


```python
#NB: Kaggle requires phone verification to use the internet or a GPU. If you haven't done that yet, the cell below will fail
#    This code is only here to check that your internet is enabled. It doesn't do anything else.
#    Here's a help thread on getting your phone number verified: https://www.kaggle.com/product-feedback/135367

import socket,warnings
try:
    socket.setdefaulttimeout(1)
    socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(('1.1.1.1', 53))
except socket.error as ex: raise Exception("STOP: No internet. Click '>|' in top right and set 'Internet' switch to on")
```


```python
#hide
!pip install -Uqq fastai
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book() 
```

    /opt/conda/lib/python3.10/site-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.23.5
      warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"



```python
#hide
from fastbook import *
from fastai.vision.widgets import *
```


```python
search_images_ddg
```




    <function fastbook.search_images_ddg(term, max_images=200)>




```python
search_images_ddg("puffer fish")
```




    (#200) ['http://4.bp.blogspot.com/-XDgL7x-fNUg/UWGMyJVfe3I/AAAAAAAABak/J-oUe9kbAcY/s1600/Dwarf+Puffer+5.jpg','https://images.pexels.com/photos/3332538/pexels-photo-3332538.jpeg?auto=compress&cs=tinysrgb&h=750&w=1260','https://i1.wp.com/rangerrick.org/wp-content/uploads/2018/04/RR-Pufferfish-Sept-2016.jpg?fit=1156%2C650&ssl=1','https://secure.i.telegraph.co.uk/multimedia/archive/03084/puffer_fish_3084634k.jpg','http://stockarch.com/files/12/12/pufferfish.jpg','http://3.bp.blogspot.com/-jOgR2bH8OPo/UTQO7NNc6XI/AAAAAAAAAsI/mX29-gxNK_k/s1600/Pufferfish.jpg','http://wwwchem.uwimona.edu.jm/courses/CHEM2402/Crime/pufferfish.jpg','https://fthmb.tqn.com/83PGRHlPECO11q3FeKNTP_A5uYY=/2126x1413/filters:fill(auto,1)/dv511069-56a81f185f9b58b7d0f0dbac.jpg','https://otlibrary.com/wp-content/gallery/puffer-fish/33.jpg','http://2.bp.blogspot.com/-DEmIWhLp6A0/T-DEbN2RUfI/AAAAAAAAAKs/QfItsQ0MdXg/s1600/northern+puffer+fish.JPG'...]




```python
results = search_images_ddg('puffer fish')
len(results)
```




    200




```python
results[0]
```




    'http://4.bp.blogspot.com/-XDgL7x-fNUg/UWGMyJVfe3I/AAAAAAAABak/J-oUe9kbAcY/s1600/Dwarf+Puffer+5.jpg'




```python
dest = 'images/puffer.jpg'
download_url(results[0], dest)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





<div>
  <progress value='245760' class='' max='244714' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.43% [245760/244714 00:00&lt;00:00]
</div>






    Path('images/puffer.jpg')




```python
im = Image.open(dest)
im.to_thumb(128,128)
```




    
![png](/images/output_9_0.png)
    




```python
poisonous_fish_types = 'puffer fish','Red Lionfish','Candiru fish', 'Great White Shark', 'Moray Eel', 'Tigerfish', 'Piranha fish', 'Stonefish', 'Atlantic Manta', 'Electric Eel'
path = Path('poisonous')
```


```python
if not path.exists():
    path.mkdir()
    for o in poisonous_fish_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_ddg(f'{o}')
        download_images(dest, urls=results)    
```


```python
fns = get_image_files(path)
fns
```




    (#1934) [Path('poisonous/Moray Eel/31c32b17-41a6-4abe-afb4-a8e41c4d087f.jpg'),Path('poisonous/Moray Eel/9b6b5292-8448-4597-9130-e099a60a3cd7.jpg'),Path('poisonous/Moray Eel/0400a40b-7429-402a-a87d-d0c10df0044f.jpg'),Path('poisonous/Moray Eel/31710fc7-c337-4f15-8be2-a0ea6cb44987.jpg'),Path('poisonous/Moray Eel/cd3145cb-3646-4bf8-8fff-4d5a6a5cc91b.jpg'),Path('poisonous/Moray Eel/69f98f57-2ba3-4638-8033-f4b896740cb1.jpg'),Path('poisonous/Moray Eel/5dbc6ba4-26e6-4d1a-9d9d-9b4478b6fdbc.jpg'),Path('poisonous/Moray Eel/0511c83b-250b-4c99-b611-19857ddc623e.jpg'),Path('poisonous/Moray Eel/dfc8bb55-4783-486c-aa56-e779a153eaf7.jpg'),Path('poisonous/Moray Eel/9a0a27f3-d16f-4d31-8c12-d68895fca116.jpg')...]




```python
failed = verify_images(fns)
failed
```




    (#62) [Path('poisonous/Moray Eel/87eca738-2b52-44b6-9014-e3840eb4606d.jpg'),Path('poisonous/Moray Eel/d9bf3dc1-3bce-4c90-b971-fe8f422efed1.jpg'),Path('poisonous/Moray Eel/39c7f90d-e8b1-4a89-9177-11c0dc6e657d.jpg'),Path('poisonous/Moray Eel/ff015699-7612-4eb5-b611-01aafbd45255.jpg'),Path('poisonous/Red Lionfish/69b28c2d-a8e1-4576-bc52-794608e07955.jpg'),Path('poisonous/Red Lionfish/abcd70d8-086b-4903-8bbe-ef6545903093.jpg'),Path('poisonous/Red Lionfish/1d34244f-ef6b-4a5b-a188-b1d65492fe3b.jpg'),Path('poisonous/Red Lionfish/fbe939ef-c554-451f-b166-cb21f51bd6f0.jpg'),Path('poisonous/Red Lionfish/6ba7db3f-f862-489a-886a-3ead81e5b4be.jpg'),Path('poisonous/puffer fish/1e2a8f32-f3ba-428c-aa57-0eefb40fba46.png')...]




```python
failed.map(Path.unlink);
```


```python
poisonous_fishes = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
     
```


```python
dls = poisonous_fishes.dataloaders(path)
```


```python
dls.valid.show_batch(max_n=4, nrows=1)
```


    
![png](/images/output_17_0.png)
    



```python
poisonous_fishes = poisonous_fishes.new(item_tfms=Resize(128, ResizeMethod.Squish))
dls = poisonous_fishes.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)
```


    
![png](/images/output_18_0.png)
    



```python
poisonous_fishes = poisonous_fishes.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros'))
dls = poisonous_fishes.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)
```


    
![png](/images/output_19_0.png)
    



```python
poisonous_fishes = poisonous_fishes.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))
dls = poisonous_fishes.dataloaders(path)
dls.train.show_batch(max_n=4, nrows=1, unique=True)
```


    
![png](/images/output_20_0.png)
    



```python
poisonous_fishes = poisonous_fishes.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = poisonous_fishes.dataloaders(path)
```


```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(10)
```


```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```


```python
interp.plot_top_losses(5, nrows=1)
```
