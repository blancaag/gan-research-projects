import libraries
from libraries import *

def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def plot(im):
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(im.astype(np.uint8))

def plot_ims(data, alpha=True): # plot_images(12)
    """
    Expects numpy array
    """
    nrows = 1
    if len(data.shape) == 3: ncols = 1
    else: ncols = data.shape[0] // nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(12, 5)
    fig.tight_layout()
    for i, ax in enumerate(axes.flat):
        if (data[i].shape[-1] == 4) & alpha: im = ax.imshow(data[i][:,:,[2,1,0,3]])
        else: im = ax.imshow(data[i][:,:,[2,1,0]])
        ax.set_axis_off()
        ax.title.set_visible(False)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    plt.subplots_adjust(left=0, hspace=0, wspace=0)
    plt.show()

def plot_im(data, alpha=True): # plot_images(12)
    """
    Expects numpy array
    """
    nrows = 1
    if len(data.shape) == 3: ncols = 1
    else: ncols = data.shape[0] // nrows
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)

    fig.set_size_inches(10, 10)
    fig.tight_layout()

    if (data.shape[-1] == 4) & alpha: im = ax.imshow(data[:,:,[2,1,0,3]])
    else: im = ax.imshow(data[:,:,[2,1,0]])

    ax.set_axis_off()
    ax.title.set_visible(False)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.subplots_adjust(left=0, hspace=0, wspace=0)
    plt.show()

def silence():
    back = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    return back

def speak(back): sys.stdout = back

def rotate_im(im, angle, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = im.shape[:2]

    # if the center is None, initialize it as the center of the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(im, M, (w, h))

    return rotated

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def read_pts_file(filename):
    """A helper function to read ibug .pts landmarks from a file."""
    lines = open(filename).read().splitlines()
    lines = lines[3:71]

    landmarks = []
    for l in lines:
        coords = l.split()
        landmarks.append([float(coords[0]), float(coords[1])])
    return landmarks

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
                      detected_face_images_path,
                      non_detected_face_images_path,
                      predictor_path, sample_size, plot_landmarks=False):

    for i in glob.glob(os.path.join(i_path, "*jpg"))[:sample_size]:
        print("Processing file: %s" %i)
        im = io.imread(i)
        im = cv2.imread(i) #.astype(np.float32)

        face_det = dlib.get_frontal_face_detector()
        shape_pred = dlib.shape_predictor(predictor_path)
        dets = face_det(im, 1)
        # second argument indicates the number of times we should upsample the image in order to detect more faces.

        # rotate images if no landmarks detected
        while len(dets) == 0:
            rot_degrees = 90
            for i in range(int(360//rot_degrees) + 1):
                im = rotate_im(im, rot_degrees*i)
                dets = face_det(im, 1)
            rot_degrees = -90
            for i in range(int(360//rot_degrees) + 1):
                im = rotate_im(im, rot_degrees*i)
                dets = face_det(im, 1)

        print("Number of faces detected: {}".format(len(dets)))

        f_name = i.split('/')[-1].split('.')[0]

        if len(dets) == 0:
            cv2.imwrite(os.path.join(non_detected_face_images_path, f_name + '.jpg'),  im)

        else:
            id_path = os.path.join(detected_face_images_path, f_name)

            if not os.path.exists(id_path): os.mkdir(id_path)
            else:
                shutil.rmtree(id_path)
                os.mkdir(id_path)

#             cv2.imwrite(os.path.join(id_path, f_name + '.jpg'), im)

            for _, i in enumerate(dets):
                (x0, y0, x1, y1) = i.left(), i.top(), i.right(), i.bottom();
                # get the landmarks/parts for the face in box d.
                landmarks = shape_to_np(shape_pred(im, i))
                save_landmarks_as_pts(landmarks, id_path, f_name.split('.')[0])

            if plot_landmarks:
                for _, i in enumerate(landmarks):
                    point = (i[0], i[1])
                    cv2.circle(im, point, 1, (0, 0, 255), -1)

            cv2.imwrite(os.path.join(id_path, f_name + '_landmarked.jpg'),  im)

    n_im_lanmarked = glob.glob(os.path.join(o_path, "*"))
    return n_im_lanmarked

def compute_3D_shape_and_texture(landmarks_path, path_to_eos):

    model = eos.morphablemodel.load_model(os.path.join(path_to_eos, "eos/share/sfm_shape_3448.bin"))
    blendshapes = eos.morphablemodel.load_blendshapes(os.path.join(path_to_eos, "eos/share/expression_blendshapes_3448.bin"))

    # alternative: .scm models
#     path_to_sfm = '/home/blanca/Documents/project/resources/www.cvssp.org/facemodels/shape/sfm.py'
#     sys.path.append(path_to_scm_models)
#     import sfm
#     shape_only = False

    model = eos.morphablemodel.load_model('/home/blanca/Documents/project/resources/www.cvssp.org/facemodels/shape/eos/sfm_shape_29587.bin')
    blendshapes = eos.morphablemodel.load_blendshapes('/home/blanca/Documents/project/resources/www.cvssp.org/facemodels/shape/expression/expression_blendshapes_29587.bin')
    print('Using the 30k vertex SFM model..')

    landmark_mapper = eos.core.LandmarkMapper(os.path.join(path_to_eos, 'eos/share/ibug_to_sfm.txt'))
    edge_topology = eos.morphablemodel.load_edge_topology(os.path.join(path_to_eos, 'eos/share/sfm_3448_edge_topology.json'))
    contour_landmarks = eos.fitting.ContourLandmarks.load(os.path.join(path_to_eos, 'eos/share/ibug_to_sfm.txt'))
    model_contour = eos.fitting.ModelContour.load(os.path.join(path_to_eos, 'eos/share/model_contours.json'))

    for i in glob.glob(os.path.join(landmarks_path, "*")):
        f_name = i.split('/')[-1]

        im = cv2.imread(os.path.join(i, f_name + '.jpg'))

        """Demo for running the eos fitting from Python."""
        landmarks = read_pts_file(os.path.join(i, f_name + '.pts'))
        landmark_ids = list(map(str, range(1, 69))) # generates the numbers 1 to 68, as strings
        im_width = im.shape[1] #1280 # Make sure to adjust these when using your own images!
        im_height = im.shape[0] #1024

        (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(model, blendshapes,
            landmarks, landmark_ids, landmark_mapper, im_width, im_height,
            edge_topology, contour_landmarks, model_contour)

        # Now you can use your favourite plotting/rendering library to display the fitted mesh, using the rendering
        # parameters in the 'pose' variable.

        isomap = eos.render.extract_texture(mesh, pose, im)
        cv2.imwrite(os.path.join(i, f_name + '.isomap.png'), isomap)
        eos.core.write_textured_obj(mesh, os.path.join(i, f_name + '.obj'))

    n_meshes = len(glob.glob(os.path.join(landmarks_path, "*")))
    return n_meshes


def render_profiles(input_folder, degrees_step = 10):
    # TODO: mute warning
    stdout = silence() # filters the stdout

    import menpo3d
    from mayavi import mlab

    for f in glob.glob(os.path.join(input_folder, "*")):

        f_name = f.split('/')[-1]
        output_path = input_folder
        profiles_folder_path = os.path.join(f, '%s_profiles' %f_name)
        if not os.path.exists(profiles_folder_path): os.mkdir(profiles_folder_path)

        mesh = menpo3d.io.import_mesh(os.path.join(f, f_name + '.obj'))
        mesh.view()

        s = mlab.gcf()
        s.scene.camera.view_up = np.array([0, 0, 0])

        offset_angle = -90 # angle in the x plane wrt the frontal view
        n_shots = 2 * abs(offset_angle) / degrees_step

        for i in range(n_shots):
            mlab.view(azimuth=0, elevation=offset_angle + i * degrees_step, roll=0)
            s.scene.save(os.path.join(profiles_folder_path, f_name + '_%d.jpg'%i))

        n_profiles = len(glob.glob(os.path.join(profiles_folder_path, "*")))

    speak(stdout) # unmute
    return n_profiles
