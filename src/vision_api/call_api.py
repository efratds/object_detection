from google.cloud import vision
import os
import io

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="client_secrets.json"


def load_from_path(path):
    from google.cloud import vision
    import io

    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)
    return image

def localize_objects(im):
    """Localize objects in the local image.
    Args:
    path: The path to the local file.
    """
    image = im

    objects = client.object_localization(
        image=image).localized_object_annotations

    print('Number of objects found: {}'.format(len(objects)))
    for object_ in objects:
        print('\n{} (confidence: {})'.format(object_.name, object_.score))
        print('Normalized bounding polygon vertices: ')
        for vertex in object_.bounding_poly.normalized_vertices:
            print(' - ({}, {})'.format(vertex.x, vertex.y))

def detect_labels(im):
    """Detects labels in the file."""

    image = im

    response = client.label_detection(image=image)
    labels = response.label_annotations
    print('Labels:')

    for label in labels:
        print(label.description)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

def detect_safe_search(im):
    """Detects unsafe features in the file."""

    image = im

    response = client.safe_search_detection(image=image)
    safe = response.safe_search_annotation

    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')
    print('Safe search:')

    print('adult: {}'.format(likelihood_name[safe.adult]))
    print('medical: {}'.format(likelihood_name[safe.medical]))
    print('spoofed: {}'.format(likelihood_name[safe.spoof]))
    print('violence: {}'.format(likelihood_name[safe.violence]))
    print('racy: {}'.format(likelihood_name[safe.racy]))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))


# contains the utilities for extracting image properties.
client = vision.ImageAnnotatorClient()

current_path = os.getcwd()
img_directory = os.path.abspath(os.path.join(
    current_path,
    '..',
    'test'))

for filename in os.listdir(img_directory):
    if filename.endswith(".jpeg"):
        try:
            print("<<<",filename,">>>")
            im = load_from_path((os.path.join(img_directory, filename)))
            localize_objects(im)
            detect_labels(im)
            detect_safe_search(im)
        except Exception as e:
                print("Problem in annotation request: " + str(e))

# with io.open(path, 'rb') as image_file:
#         content = image_file.read()
#
# image = vision.types.Image(content=content)
# response = client.image_properties(image=image)
# props = response.image_properties_annotation

# print('Properties of the image:')
# for color in props.dominant_colors.colors:
#     print('Fraction: {}'.format(color.pixel_fraction))
#     print('\tr: {}'.format(color.color.red))
#     print('\tg: {}'.format(color.color.green))
#     print('\tb: {}'.format(color.color.blue))