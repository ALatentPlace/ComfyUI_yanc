import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image, ImageSequence, ImageOps
from PIL.PngImagePlugin import PngInfo
import random
import folder_paths
import hashlib
import numpy as np
import os
from pathlib import Path
from comfy.cli_args import args
import json
import math


def permute_to_image(image):
    image = T.ToTensor()(image).unsqueeze(0)
    return image.permute([0, 2, 3, 1])[:, :, :, :3]


def to_binary_mask(image):
    images_sum = image.sum(axis=3)
    return torch.where(images_sum > 0, 1.0, 0.)


def print_brown(text):
    print("\033[33m" + text + "\033[0m")


def print_cyan(text):
    print("\033[96m" + text + "\033[0m")


def print_green(text):
    print("\033[92m" + text + "\033[0m")


class YANCRotateImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "rotation_angle": ("INT", {
                    "default": 0,
                    "min": -359,
                    "max": 359,
                    "step": 1,
                    "display": "number"})
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")

    FUNCTION = "do_it"

    CATEGORY = "YANC"

    def do_it(self, image, rotation_angle):
        samples = image.movedim(-1, 1)
        height, width = F.get_image_size(samples)

        rotation_angle = rotation_angle * -1
        rotated_image = F.rotate(samples, angle=rotation_angle, expand=True)

        empty_mask = Image.new('RGBA', (height, width), color=(255, 255, 255))
        rotated_mask = F.rotate(empty_mask, angle=rotation_angle, expand=True)

        img_out = rotated_image.movedim(1, -1)
        mask_out = to_binary_mask(permute_to_image(rotated_mask))

        return (img_out, mask_out)


class YANCText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    FUNCTION = "do_it"

    CATEGORY = "YANC"

    def do_it(self, text):
        return (text,)


class YANCTextCombine:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "text_append": ("STRING", {"forceInput": True}),
                "delimiter": ("STRING", {"multiline": False, "default": ", "}),
                "add_empty_line": ("BOOLEAN", {"default": False})
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    FUNCTION = "do_it"

    CATEGORY = "YANC"

    def do_it(self, text, text_append, delimiter, add_empty_line):
        if text_append.strip() == "":
            delimiter = ""

        str_list = [text, text_append]

        if add_empty_line:
            str_list = [text, "\n\n", text_append]

        return (delimiter.join(str_list),)


class YANCTextPickRandomLine:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    FUNCTION = "do_it"

    CATEGORY = "YANC"

    def do_it(self, text, seed):
        lines = text.splitlines()
        random.seed(seed)
        line = random.choice(lines)

        return (line,)


class YANCClearText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "chance": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    FUNCTION = "do_it"

    CATEGORY = "YANC"

    def do_it(self, text, chance):
        dice = random.uniform(0, 1)

        if chance > dice:
            text = ""

        return (text,)

    @classmethod
    def IS_CHANGED(s, text, chance):
        return s.do_it(s, text, chance)


class YANCTextReplace:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "find": ("STRING", {
                    "multiline": False,
                    "Default": "find"
                }),
                "replace": ("STRING", {
                    "multiline": False,
                    "Default": "replace"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    FUNCTION = "do_it"

    CATEGORY = "YANC"

    def do_it(self, text, find, replace):
        text = text.replace(find, replace)

        return (text,)


class YANCTextRandomWeights:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "min": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "round": 0.1,
                    "display": "number"}),
                "max": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "round": 0.1,
                    "display": "number"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    FUNCTION = "do_it"

    CATEGORY = "YANC"

    def do_it(self, text, min, max, seed):
        lines = text.splitlines()
        count = 0
        out = ""

        random.seed(seed)

        for line in lines:
            count += 1
            out += "({}:{})".format(line, round(random.uniform(min, max), 1)
                                    ) + (", " if count < len(lines) else "")

        return (out,)


class YANCLoadImageAndFilename:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(
            os.path.join(input_dir, f))]
        return {"required":
                {"image": (sorted(files), {"image_upload": True}),
                 "strip_extension": ("BOOLEAN", {"default": True})}
                }

    CATEGORY = "YANC"

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "FILENAME")

    FUNCTION = "do_it"

    def do_it(self, image, strip_extension):
        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path)
        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        if strip_extension:
            filename = Path(image_path).stem
        else:
            filename = Path(image_path).name

        return (output_image, output_mask, filename,)

    @classmethod
    def IS_CHANGED(s, image, strip_extension):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image, strip_extension):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True


class YANCSaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"images": ("IMAGE", ),
                 "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                 "folder": ("STRING", {"default": ""}),
                 "overwrite_warning": ("BOOLEAN", {"default": False}),
                 "include_metadata": ("BOOLEAN", {"default": True})
                 },
                "optional":
                    {"filename_opt": ("STRING", {"forceInput": True})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "do_it"

    OUTPUT_NODE = True

    CATEGORY = "YANC"

    def do_it(self, images, overwrite_warning, include_metadata, filename_opt=None, folder=None, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None,):

        if folder:
            filename_prefix += self.prefix_append
            filename_prefix = os.sep.join([folder, filename_prefix])
        else:
            filename_prefix += self.prefix_append

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])

        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata and include_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            if not filename_opt:
                filename_with_batch_num = filename.replace(
                    "%batch_num%", str(batch_number))

                counter = 1

                if os.path.exists(full_output_folder) and os.listdir(full_output_folder):
                    filtered_filenames = list(filter(
                        lambda filename: filename.startswith(
                            filename_with_batch_num + "_")
                        and filename[len(filename_with_batch_num) + 1:-4].isdigit(),
                        os.listdir(full_output_folder)
                    ))

                    if filtered_filenames:
                        max_counter = max(
                            int(filename[len(filename_with_batch_num) + 1:-4])
                            for filename in filtered_filenames
                        )
                        counter = max_counter + 1

                file = f"{filename_with_batch_num}_{counter:05}.png"
            else:
                if len(images) == 1:
                    file = f"{filename_opt}.png"
                else:
                    raise Exception(
                        "Multiple images and filename detected: Images will overwrite themselves!")

            save_path = os.path.join(full_output_folder, file)

            if os.path.exists(save_path) and overwrite_warning:
                raise Exception("Filename already exists.")
            else:
                img.save(save_path, pnginfo=metadata,
                         compress_level=self.compress_level)

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })

        return {"ui": {"images": results}}


class YANCLoadImageFromFolder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"image_folder": ("STRING", {"default": ""})
                 },
                "optional":
                    {"index": ("INT",
                               {"default": -1,
                                "min": -1,
                                "max": 0xffffffffffffffff,
                                "forceInput": True})}
                }

    CATEGORY = "YANC"

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "file_name")
    FUNCTION = "do_it"

    def do_it(self, image_folder, index=-1):

        image_path = os.path.join(
            folder_paths.get_input_directory(), image_folder)

        # Get all files in the directory
        files = os.listdir(image_path)

        # Filter out only image files
        image_files = [file for file in files if file.endswith(
            ('.jpg', '.jpeg', '.png', '.webp'))]

        if index is not -1:
            print_green("INFO: Index connected.")

            if index > len(image_files) - 1:
                index = index % len(image_files)
                print_green(
                    "INFO: Index too high, falling back to: " + str(index))

            image_file = image_files[index]
        else:
            print_green("INFO: Picking a random image.")
            image_file = random.choice(image_files)

        filename = Path(image_file).stem

        img_path = os.path.join(image_path, image_file)

        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)
        if img.mode == 'I':
            img = img.point(lambda i: i * (1 / 255))
        output_image = img.convert("RGB")
        output_image = np.array(output_image).astype(np.float32) / 255.0
        output_image = torch.from_numpy(output_image)[None,]

        return (output_image, filename)

    @classmethod
    def IS_CHANGED(s, image_folder, index):
        image_path = folder_paths.get_input_directory()
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()


class YANCIntToText:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"int": ("INT",
                         {"default": 0,
                          "min": 0,
                          "max": 0xffffffffffffffff,
                          "forceInput": True}),
                 "leading_zeros": ("BOOLEAN", {"default": False}),
                 "length": ("INT",
                            {"default": 5,
                             "min": 0,
                             "max": 5})
                 }
                }

    CATEGORY = "YANC"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "do_it"

    def do_it(self, int, leading_zeros, length):

        text = str(int)

        if leading_zeros:
            text = text.zfill(length)

        return (text,)


class YANCInt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"seed": ("INT", {"default": 0, "min": 0,
                          "max": 0xffffffffffffffff}), }
                }

    CATEGORY = "YANC"

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "do_it"

    def do_it(self, seed):

        return (seed,)


class YANCFloatToInt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"float": ("FLOAT", {"forceInput": True}),
                 "function": (["round", "floor", "ceil"],)
                 }
                }

    CATEGORY = "YANC"

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "do_it"

    def do_it(self, float, function):

        result = round(float)

        if function == "floor":
            result = math.floor(float)
        elif function == "ceil":
            result = math.ceil(float)

        return (int(result),)


NODE_CLASS_MAPPINGS = {
    "> Rotate Image": YANCRotateImage,
    "> Text": YANCText,
    "> Text Combine": YANCTextCombine,
    "> Text Pick Random Line": YANCTextPickRandomLine,
    "> Clear Text": YANCClearText,
    "> Text Replace": YANCTextReplace,
    "> Text Random Weights": YANCTextRandomWeights,
    "> Load Image": YANCLoadImageAndFilename,
    "> Save Image": YANCSaveImage,
    "> Load Image From Folder": YANCLoadImageFromFolder,
    "> Int to Text": YANCIntToText,
    "> Int": YANCInt,
    "> Float to Int": YANCFloatToInt
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "> Rotate Image": "😼> Rotate Image",
    "> Text": "😼> Text",
    "> Text Combine": "😼> Text Combine",
    "> Text Pick Random Line": "😼> Text Pick Random Line",
    "> Clear Text": "😼> Clear Text",
    "> Text Replace": "😼> Text Replace",
    "> Text Random Weights": "😼> Text Random Weights",
    "> Load Image": "😼> Load Image",
    "> Save Image": "😼> Save Image",
    "> Load Image From Folder": "😼> Load Image From Folder",
    "> Int to Text": "😼> Int to Text",
    "> Int": "😼> Int",
    "> Float to Int": "😼> Float to Int"
}
