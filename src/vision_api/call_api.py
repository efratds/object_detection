from google.cloud import vision
import os
import io
import pandas as pd

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="client_secrets.json"


def load_from_path(path):

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
    df = pd.DataFrame(columns=['objects_list','scores_list'])

    image = im

    objects = client.object_localization(
        image=image).localized_object_annotations

    print('Number of objects found: {}'.format(len(objects)))
    for object_ in objects:
        print('\n{} (confidence: {})'.format(object_.name, object_.score))
        print('Normalized bounding polygon vertices: ')
        for vertex in object_.bounding_poly.normalized_vertices:
            print(' - ({}, {})'.format(vertex.x, vertex.y))

    objects_list = [object_.name for object_ in objects]
    scores_list =  [object_.score for object_ in objects]

    df = df.append({'objects_list': objects_list, 'scores_list': scores_list}, ignore_index=True)

    return df

def detect_labels(im):
    """Detects labels in the file."""
    df = pd.DataFrame(columns=['labels'])

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
    labels_list =  [label_.description for label_ in labels]

    df = df.append({'labels' : labels_list }, ignore_index=True)
    return df

def detect_safe_search(im):
    """Detects unsafe features in the file."""

    df = pd.DataFrame(columns=['adult', 'medical', 'spoof', 'violence', 'racy'])

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
    df = df.append({'adult': likelihood_name[safe.adult], 'medical': likelihood_name[safe.medical],
           'spoof': likelihood_name[safe.spoof], 'violence': likelihood_name[safe.violence],
           'racy' : likelihood_name[safe.racy]}, ignore_index=True)

    return df

    #
    # def save_obj(obj, name ):
    #     with open('obj/'+ name + '.pkl', 'wb') as f:
    #         pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    #
    # def load_obj(name ):
    #     with open('obj/' + name + '.pkl', 'rb') as f:
    #         return pickle.load(f)

if __name__ == "__main__":

    data = pd.DataFrame()
    # contains the utilities for extracting image properties.
    client = vision.ImageAnnotatorClient()

    current_path = os.getcwd()
    img_directory = os.path.abspath(os.path.join(
        current_path,
        '..',
        'test'))

    for filename in os.listdir(img_directory):
        if filename.endswith(".png"):
            try:
                print("<<<",filename,">>>")
                im = load_from_path((os.path.join(img_directory, filename)))
                localize_objects = localize_objects(im)
                detect_labels = detect_labels(im)
                detect_safe_search = detect_safe_search(im)

                tmp = pd.concat([localize_objects, detect_labels, detect_safe_search], axis=1, sort=False)
                data = data.append(tmp, ignore_index=True)

            except Exception as e:
                    print("Problem in annotation request: " + str(e))

    data.to_csv("data.csv")
    print ("done")

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