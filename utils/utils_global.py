import libraries
from libraries import *

def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]

def silence():
    back = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    return back

def speak(back): sys.stdout = back

def plot(im):
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(im.astype(np.uint8))

def plot_ims(data, alpha=True, cmap=None, ncols=None, nrows=None, ): # plot_images(12)
    """
    Expects numpy array with channels last or list
    """
    if not nrows: nrows = 1
    
    if type(data) == list:
#         nrows = 1
        if not ncols: ncols = len(data)
    else:
        if len(data.shape[:1]) == 3: ncols = 1
        else: ncols = data.shape[0] // nrows
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(11, 10)
    fig.tight_layout()
    
    for i, ax in enumerate(axes.flat):
        if len(data[i].shape) == 2: 
            data_ = data[i].astype(np.uint8)
            im = ax.imshow(data_, cmap=cmap)
        elif (data[i].shape[2] == 4):
            if alpha: 
                data_ = data[i].astype(np.uint8)
                im = ax.imshow(data_[:,:,[2,1,0,3]], cmap=cmap)
            else: 
                data_ = data[i].astype(np.uint8)
                im = ax.imshow(data_[:,:,[2,1,0]], cmap=cmap)
        elif (data[i].shape[2] == 3): 
            data_ = data[i].astype(np.uint8)
            im = ax.imshow(data_[:,:,[2,1,0]], cmap=cmap)
        else: 
            print('Data dimensions not supported')
        
        ax.set_axis_off()
        ax.title.set_visible(False)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    plt.subplots_adjust(left=0, hspace=0, wspace=0)
#     plt.tight_layout()
    plt.show()
    
def plot_im(data, alpha=True, cmap=None): # plot_images(12)
    """
    Expects numpy array
    """
    nrows, ncols = 1, 1
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(10, 10)
    fig.tight_layout()

    data = data.astype(np.uint8)

    if len(data.shape) == 2: im = ax.imshow(data, cmap=cmap)
    elif (data.shape[2] == 4):
        if alpha: im = ax.imshow(data[:,:,[2,1,0,3]], cmap=cmap)
        else: im = ax.imshow(data[:,:,[2,1,0]], cmap=cmap)
    elif (data.shape[2] == 3): im = ax.imshow(data[:,:,[2,1,0]], cmap=cmap)
    else: print('Data dimensions not supported')

    ax.set_axis_off()
    ax.title.set_visible(False)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.subplots_adjust(left=0, hspace=0, wspace=0)
    plt.show()

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
    if int(lines[1:2][0].split('n_points:')[-1]) != 68:
        print ('No 68 landmark format')
        return None
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
