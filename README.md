# ROBOT CAN IMAGINE : Object detection for service robot using Generative Adverserial Newtorks 

Demonstration of RCI, an imaginative approach object detection, for service-robots, to generate images based on users speech to fetch items.

## Demonstration

1. Environment with fruits and objects placed in "living room", "bed-room","kitchen" and "Drawing-room"

![](demo/map.png)

1. User commands robot to get "Banana from living room".

2. robot understands banana fruit word through NLP and generates an image representation of banana through text to image GANs using attention mechanism.

![](architecture)

`proposed TI-GAN architecture`


3. Detect and fetch the object based from the goal location 

![](demo/demo.gif)

## Result
![](demo/results.png)

We've developed a pipeline for making an assistive robot comprehend and fetch objects based on their visual representation. The suggested TI-GANs created images based on the user's spoken command and then travelled to the intended location, where the generated images were compared to items at the site. The produced objects are akin to "imagination of things," in which the object learns attributes from a dataset and uses those features to build pictures based on the description. We did this by calculating similarity scores for picture comparison utilising the attention mechanism and residual networks in the TI-GAN architecture. For future works, this method can be used as online-incremental learning and deep reinforcement learning to learn with fewer data initially to get image representation (GANs) and continuously learn about objects to fetch from its interaction with the environment.
