from PIL import ImageDraw
from torchvision.transforms.v2.functional import to_pil_image


def render_annotated_img(img, target):
    img = to_pil_image(img)
    draw = ImageDraw.Draw(img)
    for bbox, label in zip(target.bboxes, target.class_labels):
        draw.rectangle(bbox.tolist(), outline="red")
        draw.text((bbox[0], bbox[1]), str(label.item()), fill="red")
    return img
