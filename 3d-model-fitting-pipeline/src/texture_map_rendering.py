from libraries import *
from utils import *

def render_profiles(i_path, o_path=None, degrees_step=10, sample_size=None):
    """
    i_path: path to the texture map images folder
    o_path: output path -by default create an 'isomaps' folder in 'i_path' [.obj and .isomap.png]
    degrees_step: degrees between each profile shot is taken in a range from -90 to +90 degrees wrt. a frontal position
    sample_size: if desired to compute a sample of the images instead of all the 'i_path' folders
    """

    # supported image format
    i_formats = ['**/*.isomap.png']

    # setting paths
    if not i_path: i_path = os.path.join(o_path, 'fitting')
    o_path = os.path.join(o_path, 'profiles')
    if not os.path.exists(o_path): os.makedirs(o_path)
    else: print('Warning: "%s" folder already exists: adding files..' %o_path)

    # images paths list
    i_pl = reduce(operator.add,
                  [glob(os.path.join(i_path, i), recursive=True) for i in i_formats],
                  [])
    if not sample_size: sample_size = len(i_pl)

    print('Total available images: %d' %len(i_pl))

    # TODO: mute warning

    import menpo3d
    from mayavi import mlab

    for i in i_pl[:sample_size]:

        # looping over the images
        f_name = i.split('/')[-1].split('.isomap.png')[0]
        if f_name.split('_')[-1] == 'mirror': continue

    # for f in glob.glob(os.path.join(input_folder, "*")):
    #
    #     f_name = f.split('/')[-1]
    #     output_path = input_folder
    #     profiles_folder_path = os.path.join(f, '%s_profiles' %f_name)
    #     if not os.path.exists(profiles_folder_path): os.mkdir(profiles_folder_path)

        # skip if already exists
        if len(glob(os.path.join(o_path, f_name + '.png'))) > 0:
            print(f_name, "already computed")
            continue

        for j in ['.obj', '.mtl']:
            if len(os.path.join(i_path, f_name + j)) == 0:
                print("No %s file detected for UV map: %s" %(j, i))
                continue

        stdout = silence() # filters the stdout

        mesh = menpo3d.io.import_mesh(os.path.join(i_p, f_name + '.obj'))
        mesh.view()

        s = mlab.gcf()
        s.scene.camera.view_up = np.array([0, 0, 0])

        offset_angle = -90 # angle in the x plane wrt the frontal view
        n_shots = 2 * abs(offset_angle) / degrees_step

        for k in range(n_shots):
            mlab.view(azimuth=0, elevation=offset_angle + k * degrees_step, roll=0)
            s.scene.save(os.path.join(o_path, f_name + '_%d.png'%i))

        n_profiles = len(glob.glob(os.path.join(o_path, "*.png")))

    speak(stdout) # unmute
    return n_profiles
