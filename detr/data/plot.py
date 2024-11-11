from PIL import ImageDraw
from torchvision.transforms.v2.functional import to_pil_image

from .transforms import resize_bboxes


def render_annotated_img(img, bboxes):
    pil_img = to_pil_image(img)
    draw = ImageDraw.Draw(pil_img)
    if bboxes.ndim > 1:
        bboxes = bboxes.flatten()
    if bboxes.bboxes.mean() < 1:
        h, w = (img.shape[-2], img.shape[-1])
        bboxes = resize_bboxes(bboxes, (1, 1), (h, w))

    for bbox in bboxes:
        draw.rectangle(bbox.bboxes.tolist(), outline="red")
        if hasattr(bbox, "class_labels"):
            draw.text((bbox.bboxes[0], bbox.bboxes[1]), str(bbox.class_labels.item()), fill="red")
    return pil_img
