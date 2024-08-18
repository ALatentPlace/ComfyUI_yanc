# :smirk_cat: YANC- Yet Another Node Collection

This is another node collection for ComfyUI. It includes some basic nodes that I find useful, and I've also created them to meet my personal needs.

## Latest Updates

**2024/08/18**: Added a small post processing suite.

**2024/06/10**: Added RGBColor, GetMeanColor and NormalMapLighting.

**2024/05/27**: Added LightSourceMask and mask support for the rescale image node.

**2024/05/17**: Added NIKSampler and "Noise From Image"

**2024/04/14**: Added the "Resolution by Aspect Ratio" node.

**2024/04/08**: Added an option to warn the user if files will get overwritten. Added a "Scale to Image Side" node.

**2024/04/04**: Added nodes: "Int to Text", "Int", "Float to Int".

**2024/04/03**: First Commit.

## Installation

Download or clone this repository into your `ComfyUI/custom_nodes/` directory or use the ComfyUI Manager's "Install via Git URL" functionality.  That's it!

Please keep your ComfyUI up to date to ensure that everything is working well.

## Video Tutorial

Not yet done, but you can have a look at my German ComfyUI tutorial channel here: [A Latent Place][youtubelink]

You can also find me on [Discord][discordlink].

## Nodes Overview

### Image

**Rotate Image**: Rotates an image and outputs the rotated image and a mask. Use ImageCompositeMasked (ComfyUI vanilla node) to combine it with another image.

**Scale Image to Side**: Scales an image to the selected side (width, height, shortest, longest). Let's you apply a modulo if needed. When applying a mask the mask will also be resized.

**Resolution By Aspect Ration**: Based on the input image, the node calculates the aspect ratio of it and return the closest matching resolution for either SD 1.5 or SDXL.

**Load Image**: Basically the same like the ComfyUI vanilla node, but with a filename output. You can chose to strip or keep the file extension.

**Save Image**: For saving images you can additionally specify a target folder. This folder will be created inside your output directory. You also can decide to include the metadata (like the workflow) in your image. Placeholder:

| Placeholder | Replacement|
|------|------|
| %d | Day |
| %m | Month |
| %Y | Year long |
| %y | Year short |
| %H | Hour 00 - 23 |
| %I | Hour 00 - 12 |
| %p | AM/PM |
| %M | Minute |
| %S | Second |

**IMPORTANT**: When connecting the filename_opt input (which is optional) you need to know, that the filename is no longer dynamically generated. This means your images will be overwritten as long as the filename_opt does not change!

**Load Image From Folder**: Loads randomly or iterative an image from the specified folder. If "image_folder" is kept empty, the node will load a random image from the `input` directory. If a "image_folder" is specified, this folder must be present inside of the `input` directory. By connecting a primitive to the "index" input and setting it to value 0 and increment, the node will iterate through the images in the specified folder. In order to reset the index put the primitive then back to 0.

**Noise From Image**: Generates noise from an image. Can be used in combination with the NIKSampler. Parameter Description:
| Parameter | Description |
|-|-|
| magnitude | The "waviness" of the noise. |
| smoothness | The smoothness of the noise. Lower values give a more grainy result|
| noise_intensity | Intensity of the additional noise. The "amount" of dots to be created. |
| noise_resize_factor | The size of the noise dots. 2 is good for SD 1.5, while SDXL would take 3-4. |
| noise_blend_rate | The blending rate of the noise dots on the base noise. 0.15 - 0.2 is a good value. If this is set to 0, the additional noise will not be applied. |
| saturation_correction | To raise or lower the saturation of the noise. If colors seem to be washed out in the final image, try to set it higher. |
| blend_mode | Used to blend over batched images. Switched to "off" the node will create noise for every batched image sent in. |
| blend_rate | The blend intensity of batched images. Only works in combination with the blend_mode. Batched images are blended one after the other, so first image 2 on image 1, then image 3 on the result of the first blending (and so on). |

**Normal Map Lighting**: When providing a diffuse map, a normal map and a specular map you can change the lighting of an image. Tutorial available on my YouTube channel.

### Text

**Text**: A simple multiline text node.

**Text Combine**: Combines two text inputs. The "delimiter" is optional. "add_empty_line" adds an empty line between "text" and "text_append".

**Text Pick Random Line**: Picks a random line from a multiline text input. Set the seed to fixed if you want to stop the random picking.

**Clear Text**: Empties the text by the given chance (0.0 = 0%, 1.0 = 100%)

**Text Replace**: From the given text input it replaces the "find" text with the "replace" text.

**Text Random Weights**: Takes text separated by a new line from a multiline text node and adds random weights to each of the lines. The output will be concatenated, delimited with ",". You can specify the min and max values of the weights which are randomly applied. To stop the random assignement you can set the seed to fixed.

**Int to Text**: Converts an integer to a text output. By enabling "leading_zero" you can specify with "length" the length of the ouput string. Example output: 00010.

### Basic

**Int**: A simple integer output node.

**Float to Int**: Converts a float value into an integer value. Functions are: round, floor, ceil.

### Sampling

**NIKSampler**: For a "how to" on the usage please see this video (German, English subtitles):

[![Noise Injection Sampler](https://img.youtube.com/vi/59-3RZknRgk/hqdefault.jpg)](https://youtu.be/59-3RZknRgk)

### Masking

**Light Source Mask**: A node which takes the brightest parts of an image (can be adjusted with a threshold) and creates a blurred mask.

### Utils

**Get Mean Color**: Gets the mean color of the provided image. By applying a mask you can chose the area of which you want to get the mean color value.

**RGB Color**: A helper node to create RGB colors. Outputs an INT value for the color as well as for each channel. Also export the color in HEX format.

### Post Processing

**Brightness**: Set the brightness of an image.

**Contrast**: Set the contrast of an image.

**Saturation**: Set the saturation of an image.

**Sharpen**: A simple image sharpening node.

**Divide Channels**: Divide the RGB channels of an image. Returns every channel as an image. Good if you want to manipulaute only one channel of the image.

**Combine Channels**: Combines the RGB channels back to one image.

**Edge Enhance**: A simple edge enhacing node.

**Bloom**: A bloom effect.

**Blur**: A simple image blur node.

**Vignette**: Creates either a black or a white vignette on the image.

**Fog**: Creates a foggy effect in the effect. You can use a depth mask to control the fog in the back and the front.

**Scanlines**: A simple node to create a scanline effect.

**RGB Shift**: Shifts the selected RGB channel by given pixels. Can be used, for example, to create the channel shift effect of old VHS movies.

**Film Grain**: A simple node which creates a film grain effect.

**HUE**: A simple HUE shifting node.

**Lens Distorion**: To create a fisheye effect. Subject to change.

### Experimental

**Layer Weights (for IPAMS)**: Made for the IPAdapter Mad Scientist. Input form for the layer weights.

## Demo Workflows

Demo workflows can be found in the examples folder.

## Credits

- [ComfyUI][comfyuilink]


[youtubelink]: https://youtube.com/@alatentplace
[discordlink]: https://discord.gg/WWsZSnWr89
[comfyuilink]: https://github.com/comfyanonymous/ComfyUI
