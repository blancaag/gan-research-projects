import sys
sys.path.append('../../../utils')
import libraries
import utils
from utils import *
from parser import *
from compute_landmarks import *
from compute_3D_shape_and_texture import *

def main():

    # parse options
    opt = BaseOptions().parse()
    
    # setting GLOBAL INPUT/OUTPUT dirs
    
    gi_p = opt.i
    go_p = opt.o
    l_p = opt.l
    m = opt.mirror
        
    print('INPUT/OUTPUT dirs: %s   |   %s' %(gi_p, go_p))

    f_pl = [gi_p]
    
    if opt.s: 
        f_pl = [i for i in os.listdir(gi_p) if os.path.isdir(os.path.join(gi_p, i))]

    if opt.ss: 
        f_pl = [os.path.join(i, j) for i in f_pl for j in os.listdir(os.path.join(gi_p, i))] 
    
    for f_p in f_pl:
        i_p = os.path.join(gi_p, f_p)
        o_p = os.path.join(go_p, f_p)
        
        print('Computing subfolder %s @ %s' %(i_p, o_p))
        if opt.compLan:
            print('Computing landmarks..')
            compute_landmarks(i_p, o_p, mirror=m, zoom=opt.zoom, sample_size=opt.sample_size)

        if opt.comp3D:
            print('Fitting the model..')
            compute_3D_shape_and_texture(None, o_p, l_p, mirror=m)

        if opt.rendPos:
            print('Rendering poses..')
            n_profiles = render_profiles(o_p, degrees_step = opt.step_size)

if __name__ == "__main__":
    main()
    # print(sys.argv)
    # if len(sys.argv) == 1: print('At least one argument is required')
    # elif len(sys.argv) == 2: main(sys.argv[1])
    # elif len(sys.argv) == 3: main(sys.argv[1], sys.argv[2])
    # elif len(sys.argv) == 4: main(sys.argv[1], sys.argv[2], sys.argv[3])
    # else:  print('Only three arguments are accepted')
