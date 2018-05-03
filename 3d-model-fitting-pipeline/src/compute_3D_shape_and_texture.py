# from libraries import *
# from utils import *
import sys
sys.path.append('../../../utils')
import libraries
import utils
from utils import *

def read_pts_file(filename):
    """A helper function to read ibug .pts landmarks from a file."""
    lines = open(filename).read().splitlines()
    if int(lines[1:2][0].split('n_points:')[-1]) != 68:
        print ('No 68-landmark format founded')
        return None
    lines = lines[3:71]

    landmarks = []
    for l in lines:
        coords = l.split()
        landmarks.append([float(coords[0]), float(coords[1])])
    return landmarks

def compute_3D_shape_and_texture(i_path,
                                 o_path,
                                 lns_path=None,
                                 mirror=False,
                                 sample_size=None,
                                 rotate=False):
    """
    o_path: output path -by default create an 'isomaps' folder in 'i_path' [.obj and .isomap.png]                             
    i_path: path to the images folder
    lns_path: path to the corresponding landmarks with same name and extension .pts
    sample_size: if desired to compute a sample of the images instead of all the 'i_path' folders
    """

    # supported image format
    EXT_RECURSIVE = ['**/*.jpg', '**/*.JPG', '**/*.png']
    EXT = ['*.jpg', '*.JPG', '*.png']

    # setting paths
    if not i_path: i_p = os.path.join(o_path, 'landmarks/face_detected_images'); print(i_p)
    o_p = os.path.join(o_path, 'fitting'); print(o_p)
    if not lns_path: lns_p = os.path.join(o_path, 'landmarks'); print(lns_p)
    eos_path = '../resources/libraries'
    models_path = '../resources/models/facemodels'

    # currently using the 29587 vertex morphable model -change here if desired to use another one
    model = eos.morphablemodel.load_model(
    os.path.join(models_path, 'shape/eos/sfm_shape_29587.bin'))
    blendshapes = eos.morphablemodel.load_blendshapes(
    os.path.join(models_path, 'shape/expression/expression_blendshapes_29587.bin'))

    landmark_mapper = eos.core.LandmarkMapper(os.path.join(eos_path, 'eos/share/ibug_to_sfm.txt'))
    edge_topology = eos.morphablemodel.load_edge_topology(os.path.join(models_path, 'shape/eos/sfm_29587_edge_topology.json'))
    # edge_topology = eos.morphablemodel.load_edge_topology(os.path.join(eos_path, 'eos/share/sfm_3448_edge_topology.json'))
    contour_landmarks = eos.fitting.ContourLandmarks.load(os.path.join(eos_path, 'eos/share/ibug_to_sfm.txt'))
    model_contour = eos.fitting.ModelContour.load(os.path.join(eos_path, 'eos/share/model_contours.json'))

    if not os.path.exists(o_p): os.makedirs(o_p)
    else: print('Warning: "%s" folder already exists: adding files..' %o_p)

    # images paths list
    i_pl = reduce(operator.add,
                  [glob(os.path.join(i_p, i), recursive=True) for i in EXT],
                  [])
    if not sample_size: sample_size = len(i_pl)

    print('Total available images: %d' %len(i_pl))

    for i in i_pl[:sample_size]:

        # looping over the images
        f_name = i.split('/')[-1].split('.png')[0]
        if f_name.split('_')[-1] == 'mirror' and not mirror: continue
        if f_name.split('_')[-1] != 'mirror' and mirror == 'only': continue

        # skip if already exists
        if len(glob(os.path.join(o_p, f_name + '.isomap.png'))) > 0: # .obj / .mtl / .isomap.png
            print(os.path.join(o_p, f_name + '.isomap.png'))
            print(i, "already computed")
            continue

        # detect landmarks file
        l_p = os.path.join(lns_p, f_name + '.pts')
        if len(glob(l_p)) == 0:
            print('No .pts detected for file %s' %f_name)
            continue

        print('Processing file: %s ' %i, end='')
        st_l = time() # start time loop
        im = cv2.imread(i)
        # if rotate: im = rotate_im(im, -90)
        landmarks = read_pts_file(l_p)
        if landmarks is None: continue
        landmark_ids = list(map(str, range(1, 69))) # generates the numbers 1 to 68, as strings
        im_width = im.shape[1]
        im_height = im.shape[0]

        # shape fitting and texture extraction
        st_f = time() # start time fitting
        (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(model, blendshapes,
                landmarks, landmark_ids, landmark_mapper, im_width, im_height,
                edge_topology, contour_landmarks, model_contour)

        isomap = eos.render.extract_texture(mesh, pose, im)
        cv2.imwrite(os.path.join(o_p, f_name + '.isomap.png'), isomap)
        eos.core.write_textured_obj(mesh, os.path.join(o_p, f_name + '.obj'))

        print("---loop: %ss fitting: %ss---" % (time() - st_l, time() - st_f))

    n_im = len(glob(os.path.join(o_p, '*.isomap.png')))
    print('Total available images: %d | Total fitted images: %d' %(len(i_pl), n_im))
