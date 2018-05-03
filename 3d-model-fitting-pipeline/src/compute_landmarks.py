import sys
sys.path.append('../../../utils')
import libraries
import utils
from utils import *
import traceback
import logging

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def save_landmarks_as_pts(landmarks, path, f_name):
    # write data in a file.
    file_w = open(os.path.join(path, '{}.pts'.format(f_name)), 'w')

    header = ['version: 1 \n', \
              'n_points:  68 \n', \
              '{ \n']
    text = landmarks
    footer = ['}']

    file_w.writelines(header)
    file_w.writelines([str(i[0]) + ' ' + str(i[1]) + '\n' for i in text])
    file_w.writelines(footer)
    file_w.close() #to change file access modes

def compute_landmarks(i_path,
                      o_path,
                      mirror=False,
                      plot_landmarks=False,
                      zoom=1,
                      sample_size=None):

    # supported image format
    EXT_RECURSIVE = ['**/*.jpg', '**/*.JPG', '**/*.png', '**/*.ppm']
    EXT = ['.jpg', '.JPG', '.png', '.ppm']

    # predictor path
    pred_path = os.path.join('../resources/models/dlib', 'shape_predictor_68_face_landmarks.dat')

    # create folders
    o_p = os.path.join(o_path, 'landmarks')
    no_face_detected_images_p = os.path.join(o_p, 'no_face_detected_images')
    face_detected_images_p = os.path.join(o_p, 'face_detected_images')
    
    if not os.path.exists(o_p): os.makedirs(o_p)
    if not os.path.exists(no_face_detected_images_p): os.makedirs(no_face_detected_images_p)
    if not os.path.exists(face_detected_images_p): os.makedirs(face_detected_images_p)

    else: print('Warning: "%s" folder already exists: adding files..' %o_p)

    # images paths list
    i_pl = reduce(operator.add,
                  [glob(os.path.join(i_path, i), recursive=True) for i in EXT_RECURSIVE],
                  [])
    if not sample_size: sample_size = len(i_pl)

    print('Total available images: %d' %len(i_pl))

    for i in i_pl[:sample_size]:

        if i_path in ['../../../datasets/FDDB/originalPics']:
            f_name = i.split('/')[-5] + i.split('/')[-4] + i.split('/')[-3] + i.split('/')[-2] + i.split('/')[-1].split('.')[0]
        else: f_name = i.split('/')[-1].split('.jpg')[0]

        if f_name.split('_')[-1] == 'mirror' and not mirror: continue
        if f_name.split('_')[-1] != 'mirror' and mirror == 'only': continue
                                         
        # skip if already exists
        if len(glob(os.path.join(o_p, f_name + '.pts'))) > 0 or \
           len(reduce(operator.add, [glob(os.path.join(no_face_detected_images_p, f_name) + i) for i in EXT], [])) > 0: 
               print(f_name, "already computed")
               continue

        print('Processing file: %s @ zoom %d' %(i, zoom))
        
        # https://stackoverflow.com/questions/43185605/how-do-i-read-an-image-from-a-path-with-unicode-characters
        # im = cv2.imdecode(np.asarray(bytearray(open(i, 'br').read()), dtype=np.uint8),
        #                   cv2.IMREAD_UNCHANGED)

        im = cv2.imread(i, cv2.IMREAD_UNCHANGED)

        face_det = dlib.get_frontal_face_detector()
        shape_pred = dlib.shape_predictor(pred_path)
        try: dets = face_det(im, zoom)
        # second argument indicates the number of times
        # we should upsample the image in order to detect more faces.
        except Exception as e: logging.error(traceback.format_exc())

        if len(dets) == 0: 
            rotate = True
            while rotate:
                for r in range(4):
                    im = np.rot90(im, 1, axes=(1, 0))
                    dets = face_det(im, zoom)
                    if len(dets) != 0: 
                        rotate = False
                        print('Landamarks found after %d rotations' %(r+1))
                        break
                    else: print('No-landmarks found after %d rotations' %(r+1))
                break
                              
        # second argument indicates the number of times
        # we should upsample the image in order to detect more faces.
        
        if len(dets) == 0:
            print('No faces detected for file: %s' %i)
            cv2.imwrite(os.path.join(no_face_detected_images_p, f_name + '.png'),  im)
        
        else:
            for _, j in enumerate(dets):
                (x0, y0, x1, y1) = j.left(), j.top(), j.right(), j.bottom();
                # get the landmarks/parts for the face in box
                landmarks = shape_to_np(shape_pred(im, j))
                save_landmarks_as_pts(landmarks, o_p, f_name)
                
                # save (rotated) image
                cv2.imwrite(os.path.join(face_detected_images_p, f_name + '.png'),  im) 
                
                # test
                im = cv2.imread(os.path.join(face_detected_images_p, f_name + '.png'))
                landmarks = read_pts_file(os.path.join(o_p, f_name + '.pts')); #print(landmarks)
                if plot_landmarks:
                    for _, k in enumerate(landmarks):
                        point = (int(k[0]), int(k[1]))
                        cv2.circle(im, point, 1, (0, 0, 255), -1)
                        cv2.imwrite(os.path.join(o_p, f_name + '_landmarked.png'),  im)

    n_im = len(glob(os.path.join(o_p, '*.pts')))
    print('Total computed landmarks files: %d' %n_im)
